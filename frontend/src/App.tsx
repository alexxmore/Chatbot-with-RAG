import { Link, Route, Routes, useLocation } from "react-router-dom";
import Chat from "./pages/Chat";
import Admin from "./pages/Admin";

export default function App() {
  const { pathname } = useLocation();

  return (
    <div className="app">
      <header className="header">
        <span className="header-logo">💬 RAG Support Chatbot</span>
        <nav className="header-nav">
          <Link className={pathname === "/" ? "nav-link active" : "nav-link"} to="/">
            Чат
          </Link>
          <Link className={pathname === "/admin" ? "nav-link active" : "nav-link"} to="/admin">
            Адмін
          </Link>
        </nav>
      </header>

      <main className="main">
        <Routes>
          <Route path="/" element={<Chat />} />
          <Route path="/admin" element={<Admin />} />
        </Routes>
      </main>
    </div>
  );
}
