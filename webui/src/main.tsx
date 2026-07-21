import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/open-sans/index.css";
import "@fontsource-variable/noto-sans-kr/index.css";
import "./styles/global.css";
import App from "./App.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
