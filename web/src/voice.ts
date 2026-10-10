// Voice prompts: pre-generated MP3s (scripts/generate_voice.py), browser speech as a fallback.
// iPhones only play audio started by a tap, so the first Play tap "unlocks" one shared <audio>
// element with a short silent clip, and every later prompt reuses that element.
import { voiceText } from "./i18n";

export class VoicePrompts {
  private audio = new Audio();
  private unlocked = false;
  enabled = true;

  unlock() {
    if (this.unlocked) return;
    this.audio.src = "/voice/en/silence.mp3";
    this.audio.play().then(() => (this.unlocked = true)).catch(() => undefined);
  }

  play(key: string) {
    if (!this.enabled) return;
    this.audio.pause();
    this.audio.src = `/voice/en/${key}.mp3`;
    this.audio.play().catch(() => this.speak(voiceText(key)));
  }

  stop() {
    this.audio.pause();
    window.speechSynthesis?.cancel();
  }

  private speak(text: string) {
    if (!("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US";
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  }
}
