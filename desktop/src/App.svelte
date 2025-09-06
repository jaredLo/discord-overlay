<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import { ApiClient } from './lib/api'
  import DOMPurify from 'dompurify'

  const envApi = (import.meta as any)?.env?.VITE_API_BASE_URL || (import.meta as any)?.env?.VITE_API_URL
  let apiBaseUrl = (envApi && String(envApi)) || localStorage.getItem('apiBaseUrl') || 'http://127.0.0.1:8201'
  let client = new ApiClient(apiBaseUrl)

  let transcriptHtml = ''
  let waveData: number[] = []
  let transcriptEl: HTMLDivElement | null = null
  let followTail = true
  let atBottom = true
  let health: any = null
  let showScrollArrow = false
  let waveCanvas: HTMLCanvasElement | null = null
  let waveCtx: CanvasRenderingContext2D | null = null

  function saveUrl() {
    localStorage.setItem('apiBaseUrl', apiBaseUrl)
    client = new ApiClient(apiBaseUrl)
  }

  function updateScrollState() {
    const el = transcriptEl
    if (!el) return
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    atBottom = distanceFromBottom <= 8
    followTail = atBottom && !hasSelection()
    showScrollArrow = distanceFromBottom > (el.clientHeight * 0.2)
  }

  function hasSelection(): boolean {
    try {
      const sel = window.getSelection()
      if (!sel) return false
      return !!(sel.rangeCount && String(sel).length > 0)
    } catch { return false }
  }

  function maybeAutoscroll() {
    if (!followTail) return
    const el = transcriptEl
    if (!el) return
    try {
      el.scrollTop = el.scrollHeight
    } catch {}
  }

  function drawWave() {
    const canvas = waveCanvas
    if (!canvas) return
    if (!waveCtx) waveCtx = canvas.getContext('2d')
    const ctx = waveCtx
    if (!ctx) return
    const W = canvas.width = canvas.clientWidth
    const H = canvas.height = canvas.clientHeight
    ctx.clearRect(0, 0, W, H)
    // background
    ctx.fillStyle = 'rgba(24,24,24,0.20)'
    ctx.fillRect(0, 0, W, H)
    // baseline
    ctx.strokeStyle = 'rgb(136,136,136)'
    ctx.lineWidth = 1
    const mid = Math.round(H/2)
    ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke()
    const arr = waveData || []
    if (arr.length < 2) return
    const n = arr.length
    const scale = (H - 8) / 2 / 100.0
    ctx.strokeStyle = 'rgb(30,144,255)'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    for (let i=0;i<n;i++) {
      const x = Math.round((i * (W - 2)) / Math.max(1, n - 1)) + 1
      const y = Math.round(mid - arr[i] * scale)
      if (i === 0) { ctx.moveTo(x, y); } else { ctx.lineTo(x, y); }
    }
    ctx.stroke()
  }

  let tPoll: any = null
  let wPoll: any = null
  let hPoll: any = null
  onMount(() => {
    // Poll transcript
    const pollTranscript = async () => {
      try {
        const r = await client.overlayTranscript()
        const raw = r?.html || ''
        // only rerender if changed to avoid scroll jumps
        if (raw !== transcriptHtml) {
          transcriptHtml = raw
          // Wait next frame then autoscroll if following tail
          requestAnimationFrame(maybeAutoscroll)
        }
      } catch {}
    }
    tPoll = setInterval(pollTranscript, 300)
    pollTranscript()

    // Poll waveform
    const pollWave = async () => {
      try {
        const r = await client.overlayWaveform()
        const data = r?.data || []
        waveData = data
        drawWave()
      } catch {}
    }
    wPoll = setInterval(pollWave, 120)
    pollWave()

    const pollHealth = async () => {
      try { health = await client.health() } catch {}
    }
    hPoll = setInterval(pollHealth, 2000)
    pollHealth()
  })
  onDestroy(() => { clearInterval(tPoll); clearInterval(wPoll); clearInterval(hPoll) })

  function trustHtml(html: string): string {
    try { return DOMPurify.sanitize(html, { ADD_ATTR: ['style'] }) } catch { return html }
  }

  function scrollToBottom() {
    const el = transcriptEl
    if (!el) return
    try { el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' }) } catch { el.scrollTop = el.scrollHeight }
  }
</script>

<div class="container">
  <div class="row" style="gap:8px; align-items:center;">
    <div class="muted">API:</div>
    <input style="flex:1; min-width: 200px;" bind:value={apiBaseUrl} on:change={saveUrl} />
    <button on:click={() => { client = new ApiClient(apiBaseUrl); }}>Apply</button>
    {#if health}
      <div class="muted" style="margin-left:8px;">{health.listener_running ? 'listener: on' : 'listener: off'}</div>
    {/if}
  </div>

  <div class="panel transcript" bind:this={transcriptEl} on:scroll={updateScrollState} on:mouseup={updateScrollState}>
    <div class="content" on:dblclick={scrollToBottom}>
      {@html trustHtml(transcriptHtml)}
    </div>
  </div>

  <div class="panel wave">
    <canvas bind:this={waveCanvas} on:resize={drawWave} on:mousedown={drawWave}></canvas>
  </div>

  {#if showScrollArrow}
    <div class="scroll-arrow" on:click={scrollToBottom}>▼ New</div>
  {/if}
</div>
