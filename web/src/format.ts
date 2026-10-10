// Turning report data into short English sentences (templates live in config/i18n/en.json).
import { hasKey, signName, t } from "./i18n";
import type { LessonEvent } from "./types";

export function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function eventTitle(e: LessonEvent): string {
  return t(e.title_key, { ...e.detail, sign_name: signName(e.sign_code, e.value) });
}

/** One explanatory line under the title (empty when the title says it all). */
export function eventDetail(e: LessonEvent): string {
  const d = e.detail as Record<string, unknown>;
  const parts: string[] = [];
  if (e.type === "speeding" || e.type === "possible_speeding") parts.push(t("detail.speeding", d));
  else if (e.type === "info" && e.title_key === "event.stop_not_checked") parts.push(t("detail.stop_not_checked"));
  else if (e.type === "turn_ban_complied" && d.ended_by) parts.push(t(`detail.ended_by.${d.ended_by}`));
  else if (hasKey(`detail.${e.type}`)) parts.push(t(`detail.${e.type}`, d));
  if (typeof d.reason === "string" && hasKey(`detail.reason.${d.reason}`)) parts.push(t(`detail.reason.${d.reason}`, d));
  return parts.join(" ");
}

export const SEVERITY_COLOR: Record<string, string> = {
  info: "bg-slate-400", violation: "bg-red-600", possible: "bg-amber-500", complied: "bg-green-600",
};
export const SEVERITY_PILL: Record<string, string> = {
  info: "bg-slate-100 text-slate-700", violation: "bg-red-100 text-red-800",
  possible: "bg-amber-100 text-amber-900", complied: "bg-green-100 text-green-800",
};
