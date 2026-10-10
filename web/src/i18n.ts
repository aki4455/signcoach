// UI strings come from config/i18n/en.json (shared with the backend), never hard-coded in components.
import en from "../../config/i18n/en.json";
import catalogue from "../../config/rules/signs.json";

const strings: Record<string, string> = en;
export type SignEntry = (typeof catalogue.signs)[number];
const signs = new Map<string, SignEntry>(catalogue.signs.map((s) => [s.code, s]));

/** Text for an i18n key, with {placeholders} filled from params. */
export function t(key: string, params: Record<string, unknown> = {}): string {
  const template = strings[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (_, name: string) => String(params[name] ?? `{${name}}`));
}

export function hasKey(key: string): boolean {
  return key in strings;
}

/** Catalogue entry of a sign code (official text, English translation, source). */
export function signEntry(code: string | null): SignEntry | undefined {
  return code ? signs.get(code) : undefined;
}

/** English name of a sign, e.g. "Maximum speed limit 60". */
export function signName(code: string | null, value: number | null = null): string {
  if (!code) return "";
  const name = signs.get(code)?.name ?? code;
  return value != null ? `${name} ${value}` : name;
}

/** Text of a voice prompt (used when the MP3 cannot be played). */
export function voiceText(key: string): string {
  const limit = key.match(/^speed_limit_(\d+)$/);
  return limit ? t("voice.speed_limit", { value: limit[1] }) : t(`voice.${key}`);
}
