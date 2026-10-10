// TypeScript mirror of the data contracts in pipeline/types.py (CLAUDE.md §9). Keep in sync.

export type Severity = "info" | "violation" | "possible" | "complied";
export type EventType =
  | "sign_explained" | "speeding" | "possible_speeding" | "stop_not_made" | "stop_ok"
  | "turn_ban_violation" | "turn_ban_complied" | "no_entry_possible" | "info";
export type BBox = [number, number, number, number];

export interface SignEvent {
  id: string; sign_code: string; value: number | null;
  t_first: number; t_last: number; t_pass: number; conf: number; n_frames: number;
  side: "right" | "left" | "overhead"; relevant: boolean; relevance_reason: string; conditional: boolean;
  best_frame_idx: number; best_bbox: BBox;
}

export interface MotionSample {
  t: number; speed_kmh: number | null; speed_err_kmh: number | null; source: "gps" | "video" | "none";
  quality: number; yaw_rate_dps: number | null; heading_deg: number | null; stopped: boolean;
}

export interface RuleState {
  t: number; speed_limit_kmh: number | null; limit_source: "sign" | "default" | "unknown";
  limit_sign_event_id: string | null; active_bans: string[]; mandatory: string[];
}

export interface LessonEvent {
  id: string; t: number; type: EventType; severity: Severity; sign_code: string | null; value: number | null;
  title_key: string; detail: Record<string, unknown>; clip: [number, number];
  voice_key: string | null; thumbnail: string | null; sign_event_id: string | null;
}

export interface Report {
  lesson: { id: string; student: string; label: string; created_at: string; duration_s: number; is_sample: boolean };
  media: { annotated_video_url: string; poster_url: string };
  speed_source: "gps" | "video" | "none";
  calibration: Record<string, unknown>;
  summary: { score: number; signs_passed: number; violations: number; possible: number; complied: number };
  events: LessonEvent[]; timeline: RuleState[]; motion: MotionSample[]; signs: SignEvent[];
  signs_not_met: string[]; quality_warnings: string[]; processing: Record<string, unknown>;
}

export interface JobStatus {
  status: "queued" | "processing" | "done" | "failed"; stage: string; progress: number;
  queue_position: number; error_key: string | null; lesson_id: string | null;
}

export interface LessonSummary {
  id: string; student: string; label: string; is_sample: boolean; duration_s: number | null; created_at: string;
}
