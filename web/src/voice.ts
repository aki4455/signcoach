// Voice prompts: pre-generated MP3s (scripts/generate_voice.py), browser speech as a fallback.
// iPhones only play audio started by a tap, so the first Play tap "unlocks" one shared <audio>
// element with a short silent clip, and every later prompt reuses that element.
// Prompts never cut each other off: one that comes due while another is still speaking waits and then
// follows it joined by "and" (two signs on one pole -> "Main road… and pedestrian crossing."). A prompt
// that would start more than MAX_LATE_MS late is dropped, so the voice never lags behind the video.
import { voiceText } from "./i18n";

const MAX_LATE_MS = 4000;

export class VoicePrompts {
  private audio = new Audio();
  private unlocked = false;
  private busy = false;
  private clips: string[] = [];                          // clips left of the prompt being spoken
  private current: { key: string; joined: boolean } | null = null;
  private waiting: { key: string; due: number }[] = [];  // prompts that came due while busy
  enabled = true;

  constructor() {
    this.audio.addEventListener("ended", () => {
      if (this.busy) this.playNextClip();
    });
  }

  unlock() {
    if (this.unlocked) return;
    this.audio.src = "/voice/en/silence.mp3";
    this.audio.play().then(() => (this.unlocked = true)).catch(() => undefined);
  }

  /** Playback crossed an event with this voice key. */
  play(key: string) {
    if (!this.enabled) return;
    if (this.busy) this.waiting.push({ key, due: performance.now() });
    else this.say(key, false);
  }

  /** Silence now and forget waiting prompts (voice off, a jump in the video, leaving the page). */
  stop() {
    this.busy = false;
    this.clips = [];
    this.current = null;
    this.waiting = [];
    this.audio.pause();
    window.speechSynthesis?.cancel();
  }

  private say(key: string, joined: boolean) {
    this.busy = true;
    this.current = { key, joined };
    this.clips = joined ? ["and", key] : [key];
    this.playNextClip();
  }

  private playNextClip() {
    const clip = this.clips.shift();
    if (clip === undefined) return this.finished();
    this.audio.src = `/voice/en/${clip}.mp3`;
    this.audio.play().catch(() => {
      // File missing or audio blocked: say the whole prompt with the browser's own voice instead.
      const c = this.current;
      this.clips = [];
      if (c) this.speak(`${c.joined ? "and " : ""}${voiceText(c.key)}`);
      else this.finished();
    });
  }

  private finished() {
    this.busy = false;
    this.current = null;
    const now = performance.now();
    this.waiting = this.waiting.filter((w) => now - w.due <= MAX_LATE_MS);
    const next = this.waiting.shift();
    if (next && this.enabled) this.say(next.key, true);
  }

  private speak(text: string) {
    if (!("speechSynthesis" in window)) return this.finished();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US";
    u.onend = () => this.finished();
    u.onerror = () => this.finished();
    window.speechSynthesis.speak(u);
  }
}
