import { StoredKey } from "./api";

const encoder = new TextEncoder();
const decoder = new TextDecoder();

const KDF_PARAMS = {
  name: "PBKDF2",
  hash: "SHA-256",
  iterations: 310000,
};

type EncryptedApiKey = {
  ciphertext_b64: string;
  iv_b64: string;
  kdf_salt_b64: string;
  kdf_params_json: string;
};

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return window.btoa(binary);
}

function base64ToBytes(value: string): Uint8Array<ArrayBuffer> {
  const binary = window.atob(value);
  const bytes = new Uint8Array(new ArrayBuffer(binary.length));
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

async function deriveKey(passphrase: string, salt: Uint8Array<ArrayBuffer>, kdfParams: typeof KDF_PARAMS): Promise<CryptoKey> {
  const baseKey = await window.crypto.subtle.importKey("raw", encoder.encode(passphrase), "PBKDF2", false, ["deriveKey"]);
  return window.crypto.subtle.deriveKey(
    {
      name: "PBKDF2",
      salt,
      iterations: kdfParams.iterations,
      hash: { name: kdfParams.hash },
    },
    baseKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );
}

export async function encryptApiKey(apiKey: string, passphrase: string): Promise<EncryptedApiKey> {
  const salt = window.crypto.getRandomValues(new Uint8Array(new ArrayBuffer(16)));
  const iv = window.crypto.getRandomValues(new Uint8Array(new ArrayBuffer(12)));
  const key = await deriveKey(passphrase, salt, KDF_PARAMS);
  const ciphertext = await window.crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, encoder.encode(apiKey));

  return {
    ciphertext_b64: bytesToBase64(new Uint8Array(ciphertext)),
    iv_b64: bytesToBase64(iv),
    kdf_salt_b64: bytesToBase64(salt),
    kdf_params_json: JSON.stringify(KDF_PARAMS),
  };
}

export async function decryptApiKey(stored: StoredKey, passphrase: string): Promise<string> {
  const params = JSON.parse(stored.kdf_params_json) as typeof KDF_PARAMS;
  if (params.name !== "PBKDF2") {
    throw new Error(`Unsupported KDF: ${params.name}`);
  }
  const key = await deriveKey(passphrase, base64ToBytes(stored.kdf_salt_b64), params);
  const plaintext = await window.crypto.subtle.decrypt(
    { name: "AES-GCM", iv: base64ToBytes(stored.iv_b64) },
    key,
    base64ToBytes(stored.ciphertext_b64),
  );
  return decoder.decode(plaintext);
}
