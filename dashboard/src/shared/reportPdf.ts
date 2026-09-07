// Opens a branded, print-ready view of a run's final report in a new window
// and triggers the browser's "Save as PDF". No external dependencies.

import { BASE, authHeaders } from "./api";

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function attrEsc(s: string): string {
  return esc(s).replace(/"/g, "&quot;");
}

function inline(s: string): string {
  let out = esc(s);
  out = out.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, "$1<em>$2</em>");
  out = out.replace(/`([^`]+?)`/g, "<code>$1</code>");
  out = out.replace(
    /\[([^\]]+)\]\(([^)]+)\)/g,
    '<a href="$2">$1</a>',
  );
  return out;
}

export function runArtifactUrl(caseId: string, runId: string, relPath: string): string {
  const clean = normalizeArtifactPath(relPath);
  const encoded = clean.split("/").map((part) => encodeURIComponent(part)).join("/");
  // Embedded as <img src> in the print window, so it can't carry an auth
  // header — the hosted deployment authenticates these via the session cookie.
  return `${BASE}/api/cases/${encodeURIComponent(caseId)}/artifacts/${encoded}?run_id=${encodeURIComponent(runId)}`;
}

function isExternalOrEmbedded(src: string): boolean {
  return /^(https?:|data:)/i.test(src.trim());
}

/** Map report markdown paths (relative or absolute) to a run-scoped artifact path. */
function normalizeArtifactPath(src: string): string {
  const trimmed = src.trim().replace(/^\/+/, "");
  const figuresIdx = trimmed.indexOf("figures/");
  if (figuresIdx >= 0) {
    return trimmed.slice(figuresIdx);
  }
  return trimmed;
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

async function embedReportImages(
  md: string,
  caseId: string,
  runId: string,
): Promise<string> {
  const re = /!\[([^\]]*)\]\(([^)]+)\)/g;
  const matches = [...md.matchAll(re)];
  if (!matches.length) return md;

  let out = md;
  for (const match of matches) {
    const full = match[0];
    const alt = match[1];
    const src = match[2].trim();
    if (isExternalOrEmbedded(src)) continue;

    try {
      const res = await fetch(runArtifactUrl(caseId, runId, src));
      if (!res.ok) continue;
      const dataUrl = await blobToDataUrl(await res.blob());
      out = out.replace(full, `![${alt}](${dataUrl})`);
    } catch {
      // Keep original path; browser may still load via runArtifactUrl fallback.
    }
  }
  return out;
}

function mdToHtml(md: string, caseId: string, runId: string): string {
  const lines = md.replace(/\r/g, "").split("\n");
  let html = "";
  let i = 0;
  const isBreak = (l: string) =>
    /^(#{1,6}\s|>|\||!\[|\s*[-*]\s|---)/.test(l) || /^\s*$/.test(l);
  while (i < lines.length) {
    const ln = lines[i];
    if (/^\s*$/.test(ln)) {
      i++;
      continue;
    }
    const img = ln.match(/^!\[([^\]]*)\]\(([^)]+)\)\s*$/);
    if (img) {
      const alt = img[1];
      const src = img[2].trim();
      const resolved = isExternalOrEmbedded(src)
        ? src
        : runArtifactUrl(caseId, runId, src);
      html += `<figure class="report-figure"><img src="${attrEsc(resolved)}" alt="${attrEsc(alt)}" loading="eager"/><figcaption>${esc(alt)}</figcaption></figure>`;
      i++;
      continue;
    }
    const h = ln.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      html += `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`;
      i++;
      continue;
    }
    if (/^---+\s*$/.test(ln)) {
      html += "<hr/>";
      i++;
      continue;
    }
    if (/^>\s?/.test(ln)) {
      const buf: string[] = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) {
        buf.push(lines[i].replace(/^>\s?/, ""));
        i++;
      }
      html += `<blockquote>${inline(buf.join(" "))}</blockquote>`;
      continue;
    }
    if (/^\|/.test(ln) && i + 1 < lines.length && /^\|?\s*:?-/.test(lines[i + 1])) {
      const cells = (l: string) =>
        l.split("|").slice(1, -1).map((c) => c.trim());
      const header = cells(ln);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && /^\|/.test(lines[i])) {
        rows.push(cells(lines[i]));
        i++;
      }
      html +=
        "<table><thead><tr>" +
        header.map((c) => `<th>${inline(c)}</th>`).join("") +
        "</tr></thead><tbody>" +
        rows
          .map(
            (r) => "<tr>" + r.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>",
          )
          .join("") +
        "</tbody></table>";
      continue;
    }
    if (/^\s*[-*]\s+/.test(ln)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ""));
        i++;
      }
      html += "<ul>" + items.map((it) => `<li>${inline(it)}</li>`).join("") + "</ul>";
      continue;
    }
    const buf = [ln];
    i++;
    while (i < lines.length && !isBreak(lines[i])) {
      buf.push(lines[i]);
      i++;
    }
    html += `<p>${inline(buf.join(" "))}</p>`;
  }
  return html;
}

const LOGO = `<svg viewBox="0 0 48 48" width="34" height="34" fill="none" xmlns="http://www.w3.org/2000/svg">
  <polygon points="24,4 41.3,14 41.3,34 24,44 6.7,34 6.7,14" stroke="#38bdf8" stroke-width="2.4" stroke-linejoin="round"/>
  <circle cx="16" cy="17" r="2.2" fill="#38bdf8"/><circle cx="32" cy="17" r="2.2" fill="#38bdf8"/>
  <circle cx="24" cy="33" r="2.2" fill="#38bdf8"/><circle cx="24" cy="23" r="3.6" fill="#38bdf8"/>
</svg>`;

function buildHtml(body: string, dataset: string, model: string): string {
  return `<!doctype html><html><head><meta charset="utf-8"/>
<title>${esc(dataset)} — MAADS Report</title>
<style>
  :root { --blue:#1e3a8a; --cyan:#38bdf8; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Segoe UI",system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif;
    color:var(--ink); line-height:1.55; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
  .toolbar { position:sticky; top:0; display:flex; justify-content:flex-end; gap:.5rem;
    padding:.75rem 1rem; background:#0b1220; }
  .toolbar button { background:var(--cyan); color:#0b1220; border:0; border-radius:8px;
    padding:.5rem .9rem; font-weight:700; font-size:.85rem; cursor:pointer; }
  .page { max-width:820px; margin:0 auto; padding:0 0 3rem; }
  header.brand { display:flex; align-items:center; gap:.8rem;
    background:linear-gradient(135deg,#0b1220 0%,#0f2150 55%,#1e3a8a 100%);
    color:#fff; padding:1.6rem 2rem; }
  header.brand .t1 { font-size:1.4rem; font-weight:800; letter-spacing:-.01em; }
  header.brand .t2 { font-size:.8rem; color:#9fc7f0; margin-top:.15rem; }
  .meta { display:flex; flex-wrap:wrap; gap:.5rem; padding:.9rem 2rem; background:#f1f5ff;
    border-bottom:1px solid var(--line); }
  .chip { font-size:.72rem; font-weight:600; color:var(--blue); background:#dbe7ff;
    border-radius:999px; padding:.25rem .7rem; }
  .content { padding:1.5rem 2rem; }
  .content h1 { font-size:1.5rem; color:var(--blue); margin:1.4rem 0 .6rem; }
  .content h2 { font-size:1.15rem; color:var(--blue); border-bottom:2px solid #dbe7ff;
    padding-bottom:.25rem; margin:1.5rem 0 .6rem; }
  .content h3 { font-size:1rem; color:var(--ink); margin:1.1rem 0 .4rem; }
  .content p { margin:.5rem 0; }
  .content ul { margin:.4rem 0 .4rem 1.1rem; padding:0; }
  .content li { margin:.2rem 0; }
  .content code { background:#eef2ff; color:#1e3a8a; padding:.05rem .3rem; border-radius:4px;
    font-family:ui-monospace,Consolas,monospace; font-size:.85em; }
  .content blockquote { margin:.6rem 0; padding:.5rem .9rem; background:#f1f5ff;
    border-left:3px solid var(--cyan); color:#334155; border-radius:0 8px 8px 0; }
  .content hr { border:0; border-top:1px solid var(--line); margin:1.2rem 0; }
  .report-figure { margin:1rem 0 1.25rem; break-inside:avoid; page-break-inside:avoid; }
  .report-figure img { display:block; max-width:100%; height:auto; border:1px solid var(--line);
    border-radius:8px; background:#fff; }
  .report-figure figcaption { margin-top:.35rem; font-size:.78rem; color:var(--muted); text-align:center; }
  table { border-collapse:collapse; width:100%; margin:.7rem 0; font-size:.9rem; }
  th { background:var(--blue); color:#fff; text-align:left; padding:.45rem .7rem; }
  td { border-bottom:1px solid var(--line); padding:.4rem .7rem; }
  tr:nth-child(even) td { background:#f8fafc; }
  footer { text-align:center; color:var(--muted); font-size:.72rem; padding:1.5rem; }
  @media print {
    .toolbar { display:none; }
    @page { margin:14mm; }
    header.brand { border-radius:0; }
    .report-figure, .report-figure img { break-inside:avoid; page-break-inside:avoid; }
  }
</style></head>
<body>
  <div class="toolbar"><button onclick="window.print()">↓ Save as PDF</button></div>
  <div class="page">
    <header class="brand">${LOGO}<div><div class="t1">MAADS — Final Report</div>
      <div class="t2">Multi-agent autonomous data science</div></div></header>
    <div class="meta">
      <span class="chip">Dataset: ${esc(dataset)}</span>
      <span class="chip">LLM model: ${esc(model)}</span>
    </div>
    <div class="content">${body}</div>
    <footer>Generated by MAADS · ${esc(dataset)} · ${esc(model)}</footer>
  </div>
  <script>
  function printWhenReady(){
    var imgs=Array.prototype.slice.call(document.querySelectorAll(".content img"));
    if(!imgs.length){window.print();return;}
    var pending=imgs.length;
    function done(){if(--pending===0)window.print();}
    imgs.forEach(function(img){
      if(img.complete&&img.naturalWidth>0)done();
      else{img.onload=done;img.onerror=done;}
    });
  }
  window.addEventListener("load",function(){setTimeout(printWhenReady,200);});
  </script>
</body></html>`;
}

export async function openStyledReport(
  caseId: string,
  runId: string,
  dataset: string,
  model: string,
): Promise<void> {
  const w = window.open("", "_blank");
  if (!w) {
    alert("Please allow pop-ups to download the report.");
    return;
  }
  w.document.write(
    "<p style='font-family:sans-serif;padding:2rem;color:#334155'>Preparing report…</p>",
  );
  try {
    const res = await fetch(
      `${BASE}/api/cases/${encodeURIComponent(caseId)}/final_report.md?run_id=${encodeURIComponent(runId)}`,
      { headers: authHeaders() },
    );
    if (!res.ok) throw new Error(String(res.status));
    const md = await embedReportImages(await res.text(), caseId, runId);
    w.document.open();
    w.document.write(buildHtml(mdToHtml(md, caseId, runId), dataset, model));
    w.document.close();
  } catch {
    w.document.open();
    w.document.write(
      "<p style='font-family:sans-serif;padding:2rem;color:#b91c1c'>Report not available for this run.</p>",
    );
    w.document.close();
  }
}
