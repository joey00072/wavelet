import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { initialTheme } from "./lib/theme";
import "./main.css";

document.documentElement.classList.toggle("dark", initialTheme() === "dark");

createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
