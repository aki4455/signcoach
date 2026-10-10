// Minimal client-side routing: three pages don't need a router library.
import { useEffect, useState } from "react";

export function navigate(path: string) {
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export function usePath(): string {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const onChange = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onChange);
    return () => window.removeEventListener("popstate", onChange);
  }, []);
  return path;
}

/** <a> that changes page without a full reload. */
export function linkProps(path: string) {
  return {
    href: path,
    onClick: (e: { preventDefault: () => void; metaKey: boolean; ctrlKey: boolean }) => {
      if (e.metaKey || e.ctrlKey) return;
      e.preventDefault();
      navigate(path);
    },
  };
}
