import { Navigate, Route, Routes, Link, useNavigate } from "react-router-dom";
import { LoginPage } from "./pages/Login";
import { RegisterPage } from "./pages/Register";
import { ProfilePage } from "./pages/Profile";
import { TasksPage } from "./pages/Tasks";
import { CasesPage } from "./pages/Cases";
import { getToken, logout } from "./lib/api";

function RequireAuth({ children }: { children: JSX.Element }) {
  return getToken() ? children : <Navigate to="/login" replace />;
}

function NavBar() {
  const navigate = useNavigate();
  const authed = Boolean(getToken());
  return (
    <nav className="flex items-center gap-4 border-b border-slate-800 p-4 text-sm">
      <span className="font-semibold">maads</span>
      {authed && (
        <>
          <Link to="/profile">Profile</Link>
          <Link to="/cases">Cases</Link>
          <Link to="/tasks">Tasks</Link>
          {/* Plain <a>, not <Link>: the trace dashboard is a separate SPA
              served by the backend at /dashboard, not a route in this one. */}
          <a href="/dashboard/">Dashboard</a>
          <button
            className="ml-auto text-slate-400"
            onClick={() => {
              void logout();
              navigate("/login");
            }}
          >
            Log out
          </button>
        </>
      )}
    </nav>
  );
}

export function App() {
  return (
    <div className="min-h-screen bg-slate-950">
      <NavBar />
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route
          path="/profile"
          element={
            <RequireAuth>
              <ProfilePage />
            </RequireAuth>
          }
        />
        <Route
          path="/cases"
          element={
            <RequireAuth>
              <CasesPage />
            </RequireAuth>
          }
        />
        <Route
          path="/tasks"
          element={
            <RequireAuth>
              <TasksPage />
            </RequireAuth>
          }
        />
        <Route path="*" element={<Navigate to={getToken() ? "/tasks" : "/login"} replace />} />
      </Routes>
    </div>
  );
}
