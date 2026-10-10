// UI strings come from config/i18n/en.json (shared with the backend), never hard-coded in components.
import en from "../../config/i18n/en.json";
import catalogue from "../../config/rules/signs.json";

const strings: Record<string, string> = en;
const signs = new Map(catalogue.signs.map((s) => [s.code, s]));

/** Text for an i18n key, with {placeholders} filled from params. */
export function t(key: string, params: Record<string, unknown> = {}): string {
  const template = strings[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (_, name: string) => String(params[name] ?? `{${name}}`));
}

/** English working name of a sign, e.g. "Maximum speed limit 60". */
export function signName(code: string | null, value: number | null = null): string {
  if (!code) return "";
  const name = signs.get(code)?.name ?? code;
  return value != null ? `${name} ${value}` : name;
}

export function isVerified(code: string | null): boolean {
  return code != null && signs.get(code)?.verified === true;
}
