/** Record the actual rendered game at wall-clock speed, including inference waits. */
export type CaptureFrame = {
  controller: string; seed: number; simulationSeconds: number; wallSeconds: number;
  health: number; ammo: number; kills: number; core: boolean; status: string;
  action: string; latencyMs: number; confidence?: number; decisions: number;
  escalations: number; thinking: boolean;
};

export class GameplayRecording {
  readonly canvas = document.createElement('canvas');
  private context: CanvasRenderingContext2D;
  private media: MediaRecorder;
  private chunks: Blob[] = [];
  private stopped: Promise<Blob> | null = null;

  constructor() {
    this.canvas.width = 1280; this.canvas.height = 720;
    this.context = this.canvas.getContext('2d', { alpha: false })!;
    const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
      .find(type => MediaRecorder.isTypeSupported(type));
    if (!mimeType) throw new Error('This browser cannot record WebM gameplay.');
    this.media = new MediaRecorder(this.canvas.captureStream(30), { mimeType, videoBitsPerSecond: 5_000_000 });
    this.media.ondataavailable = event => { if (event.data.size) this.chunks.push(event.data); };
    this.media.start(1000);
  }

  draw(source: HTMLCanvasElement, frame: CaptureFrame) {
    const g = this.context, w = this.canvas.width, h = this.canvas.height;
    g.fillStyle = '#101611'; g.fillRect(0, 0, w, h);
    const scale = Math.min(w / source.width, h / source.height);
    const sw = source.width * scale, sh = source.height * scale;
    g.drawImage(source, (w - sw) / 2, (h - sh) / 2, sw, sh);
    g.fillStyle = '#101710e8'; g.fillRect(0, 0, w, 75); g.fillRect(0, h - 105, w, 105);
    g.font = 'bold 22px Consolas, monospace'; g.fillStyle = '#e8a456';
    g.fillText('CINDER STATION / ' + frame.controller.toUpperCase(), 26, 32);
    g.font = '14px Consolas, monospace'; g.fillStyle = '#d2dbc6';
    g.fillText(`SEED ${frame.seed}   SIM ${frame.simulationSeconds.toFixed(1)}s   WALL ${frame.wallSeconds.toFixed(1)}s`, 26, 58);
    g.font = 'bold 25px Consolas, monospace'; g.fillStyle = '#eee8d8';
    g.fillText(`HP ${Math.ceil(frame.health)}   AMMO ${frame.ammo}   HOSTILES ${frame.kills}/6   CORE ${frame.core ? 'SECURED' : 'PENDING'}`, 26, h - 67);
    g.font = '15px Consolas, monospace'; g.fillStyle = '#b8d68b';
    g.fillText(`ACTION ${frame.action}   REQUEST ${Math.round(frame.latencyMs)} ms   CONFIDENCE ${frame.confidence === undefined ? '—' : (frame.confidence * 100).toFixed(1) + '%'}   FALLBACKS ${frame.escalations}/${frame.decisions}`, 26, h - 37);
    g.font = '11px Consolas, monospace'; g.fillStyle = '#a4af9b';
    g.fillText('LIVE CANVAS CAPTURE / NORMAL SPEED / INFERENCE WAITS RETAINED / NO AUDIO', 26, h - 14);
    if (frame.thinking || frame.status !== 'playing') {
      const message = frame.thinking ? 'DECIDING · SIMULATION HELD' : frame.status === 'won' ? 'EXTRACTED' : frame.status.toUpperCase();
      g.font = 'bold 25px Consolas, monospace';
      const width = g.measureText(message).width + 42;
      g.fillStyle = '#101710ee'; g.fillRect((w - width) / 2, 99, width, 53);
      g.fillStyle = '#e8a456'; g.fillText(message, (w - width) / 2 + 21, 134);
    }
  }

  stop(): Promise<Blob> {
    if (this.stopped) return this.stopped;
    this.stopped = new Promise((resolve, reject) => {
      this.media.onstop = () => {
        for (const track of this.media.stream.getTracks()) track.stop();
        resolve(new Blob(this.chunks, { type: this.media.mimeType }));
      };
      this.media.onerror = () => {
        for (const track of this.media.stream.getTracks()) track.stop();
        reject(new Error('Gameplay recording failed.'));
      };
      if (this.media.state === 'inactive') {
        for (const track of this.media.stream.getTracks()) track.stop();
        resolve(new Blob(this.chunks, { type: this.media.mimeType }));
      } else this.media.stop();
    });
    return this.stopped;
  }
}
