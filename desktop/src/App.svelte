<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import { ApiClient } from './lib/api'
  import DOMPurify from 'dompurify'
  import { marked } from 'marked'

  const envApi = (import.meta as any)?.env?.VITE_API_BASE_URL || (import.meta as any)?.env?.VITE_API_URL
  let apiBaseUrl = (envApi && String(envApi)) || localStorage.getItem('apiBaseUrl') || 'http://127.0.0.1:8201'
  let client = new ApiClient(apiBaseUrl)

  let transcriptHtml = ''
  let transcriptRaw = ''
  let asrBackend = ''
  let waveData: number[] = []
  let transcriptEl: HTMLDivElement | null = null
  let followTail = true
  let atBottom = true
  let showScrollArrow = false
  let waveCanvas: HTMLCanvasElement | null = null
  let waveCtx: CanvasRenderingContext2D | null = null
  let sidebarOpen = true
  const SIDEBAR_W_OPEN = 260
  const SIDEBAR_W_CLOSED = 26
  type Vocab = { 
    ja: string, 
    read?: string, 
    en: string, 
    ctx?: string,
    meanings?: string[],
    kanji_breakdown?: Array<{kanji: string, reading: string, meaning: string}>,
    word_type?: string,
    nuance?: string,
    usage_examples?: string[]
  }
  let vocabs: Vocab[] = []
  let vocabsUniq: Array<Vocab & { count: number }> = []
  let vocabsTimeline: Array<Vocab & { count: number }> = []
  // Enhanced hover bubble state
  let hoverVisible = false
  let hoverVocab: Vocab | null = null
  let hoverTop = 0
  let hoverLeft = 0
  let highlightedEl: HTMLElement | null = null
  import { appWindow } from '@tauri-apps/api/window'
  import { invoke } from '@tauri-apps/api/tauri'
  import SidebarPanel from './lib/components/SidebarPanel.svelte'
  const sessionStarted = new Date()
  let autosaveTimer: any = null
  let closeUnlisten: (() => void) | null = null
  let isClosing = false
  // Right sidebar (suggestions)
  let detailsEnabled = true
  let rightOpen = true
  const RIGHT_W_OPEN = 260
  const RIGHT_W_CLOSED = 26
  type Suggest = { ja: string, read?: string, en?: string, ctx?: string, hint?: string }
  let suggTimeline: Array<Suggest & { count: number }> = []
  const suggCounts: Record<string, number> = {}
  // Far right: Raw transcription (append-only chat-style)
  let rawEnabled = true
  let rawOpen = true
  const RAW_W_OPEN = 260
  const RAW_W_CLOSED = 26
  type RawItem = { id: string, text: string }
  let rawTimeline: RawItem[] = []
  const rawSeenIds: Record<string, boolean> = {}
  const jpCharRe = /[\u30A0-\u30FF\u3040-\u309F\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF々〆ヵヶ]/
  function isJapanese(s: string): boolean { try { return jpCharRe.test(s) } catch { return false } }
  // Far right: ASR debug
  let asrDebugEnabled = false
  let debugOpen = true
  const DEBUG_W_OPEN = 360
  const DEBUG_W_CLOSED = 26
  type AsrRow = { id: string, ts: number, openai?: { text?: string, ms?: number }, remote?: { text?: string, ms?: number } }
  let asrRows: AsrRow[] = []


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
  onMount(async () => {
    // Poll transcript
    const pollTranscript = async () => {
      try {
        const r = await client.overlayTranscript()
        const raw = (r?.text ?? r?.html ?? '')
        if (raw !== transcriptRaw) {
          transcriptRaw = raw
          // Render as Markdown with single-line breaks preserved, stripping vocab lines from main view
          const display = stripVocabSections(transcriptRaw)
          let html = marked.parse(display, { breaks: true, gfm: true }) as string
          // Post-process: wrap readings inside parentheses after colored spans
          try {
            const re = new RegExp('(<span\\\b[^>]*style=\\"[^\\"]*color:[^\\\";]+[^\\\">]*>[^<]+<\\/span>)\\(([^)、]+)([)、])', 'g')
            html = html.replace(re, (_m, s1, reading, tail) => `<span class=\"ja-inline\">${s1}</span>(<span class=\"reading-inline\">${reading}</span>${tail}`)
          } catch {}
          transcriptHtml = html || ''
          // Extract vocabs in order of appearance
          vocabs = extractVocabs(transcriptRaw)
          vocabsUniq = dedupeVocabs(vocabs)
          vocabsTimeline = accrueCounts(vocabs)
          // Raw transcription now sourced from ASR debug endpoint (handled in poll below)
          // Build right-side suggestions (exclude existing left vocabs), enrich via API once per update
          if (detailsEnabled) {
            try {
              const s = await client.suggestions()
              const items = (s.items||[]).filter(x => !vocabsUniq.find(v => v.ja === x.ja))
              for (const it of items) {
                const prev = suggCounts[it.ja] || 0
                const next = prev + 1
                suggCounts[it.ja] = next
                suggTimeline = [...suggTimeline, { ...it, count: next }]
              }
            } catch {}
          }
          // Wait next frame then autoscroll if following tail
          requestAnimationFrame(maybeAutoscroll)
        }
      } catch {}
    }
    tPoll = setInterval(pollTranscript, 300)
    pollTranscript()

    // Read health once (no gating; suggestions default on)
    try { const h = await client.health(); asrBackend = (h?.asr_backend || '').toLowerCase() } catch {}

    // Suggestions poll (append-only timeline)
    const pollSugg: any = async () => {
      if (pollSugg._busy) return; pollSugg._busy = true
      try {
        const s = await client.suggestions();
        const items = (s.items||[]).filter(x => !vocabsUniq.find(v => v.ja === x.ja))
        if (items.length) {
          for (const it of items) {
            const prev = suggCounts[it.ja] || 0
            const next = prev + 1
            suggCounts[it.ja] = next
            suggTimeline = [...suggTimeline, { ...it, count: next }]
          }
        }
      } catch {}
      finally { pollSugg._busy = false }
    }
    setInterval(pollSugg, 1000); pollSugg()

    // ASR debug poll (always used to feed Raw transcription; UI debug panel gated separately)
    const pollAsr: any = async () => {
      if (pollAsr._busy) return; pollAsr._busy = true
      try {
        const r = await client.asrDebugLog()
        const items = (r.items || []) as any as AsrRow[]
        if (items.length) {
          asrRows = items
          // Update Raw transcription from newest ASR entries
          let appended = 0
          for (const it of items) {
            const key = String(it.id || it.ts)
            if (!key || rawSeenIds[key]) continue
            const text = String((it.remote?.text || it.openai?.text || '') || '').trim()
            if (!text || !isJapanese(text)) { rawSeenIds[key] = true; continue }
            rawSeenIds[key] = true
            rawTimeline = [...rawTimeline, { id: key, text }].slice(-300)
            appended++
          }
          if (appended > 0) { requestAnimationFrame(maybeAutoscroll) }
        }
      } catch {}
      finally { pollAsr._busy = false }
    }
    setInterval(pollAsr, 1000); pollAsr()

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

    // no health polling; simplify UI
    // Periodic autosave of vocab CSV every 60s
    autosaveTimer = setInterval(exportCsvPeriodic, 60000)
    // Auto-export vocabs CSV on window close (guard against re-entry)
    closeUnlisten = await appWindow.onCloseRequested(async (e) => {
      if (isClosing) return
      isClosing = true
      e.preventDefault()
      try {
        clearInterval(autosaveTimer)
        clearInterval(tPoll)
        clearInterval(wPoll)
        await exportCsvPeriodic()
        await exportRawTranscriptOnClose()
      } catch (_) {
        // ignore
      } finally {
        if (closeUnlisten) { try { closeUnlisten() } catch (_) {} }
        await appWindow.close()
      }
    })
  })
  onDestroy(() => { clearInterval(tPoll); clearInterval(wPoll); clearInterval(autosaveTimer); if (closeUnlisten) { try { closeUnlisten() } catch (_) {} } })

  function trustHtml(html: string): string {
    try { return DOMPurify.sanitize(html, { ADD_ATTR: ['style'] }) } catch { return html }
  }

  function scrollToBottom() {
    const el = transcriptEl
    if (!el) return
    try { el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' }) } catch { el.scrollTop = el.scrollHeight }
  }

  function stripTags(s: string): string {
    try { return s.replace(/<[^>]+>/g, '') } catch { return s }
  }

  function stripVocabSections(text: string): string {
    const lines = text.split(/\r?\n/)
    const kept: string[] = []
    for (const line of lines) {
      const plain = stripTags(line).trim()
      if (/^(Vocab:|📚\s*Context Vocab:)/.test(plain)) continue
      // Also remove trailing inline 'Vocab: ...' if present on same line
      const cleaned = line.replace(/\s*Vocab:\s.*$/, '')
      kept.push(cleaned)
    }
    return kept.join('\n')
  }

  function extractVocabs(text: string): Vocab[] {
    const lines = text.split(/\r?\n/)
    const out: Vocab[] = []
    const hasKanjiOrKatakana = (s: string) => {
      // Kanji ranges: CJK Unified Ideographs + Ext-A + Compatibility Ideographs
      // Katakana ranges: standard + phonetic extensions + halfwidth
      return /[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u30A0-\u30FF\u31F0-\u31FF\uFF66-\uFF9D]/.test(s)
    }
    for (let i=0;i<lines.length;i++) {
      const lineRaw = lines[i]
      const line = stripTags(lineRaw)
      const idx = line.indexOf('Vocab:')
      if (idx === -1) continue
      let rest = line.slice(idx + 6).trim()
      if (!rest) continue
      // Collect up to two prior non-empty, non-vocab lines as context
      const ctxLines: string[] = []
      for (let j = i-1; j >= 0 && ctxLines.length < 2; j--) {
        const prev = stripTags(lines[j]).trim()
        if (!prev) continue
        if (/^(Vocab:|📚\s*Context Vocab:)/.test(prev)) continue
        ctxLines.unshift(prev)
      }
      const ctx = ctxLines.join('\n')
      // Split multiple entries by '・'
      const pieces = rest.split('・').map(s => s.trim()).filter(Boolean)
      for (const seg of pieces) {
        if (/\(none\)/i.test(seg)) continue
        // Match: JA(READ) — EN  (em dash or hyphen)
        const m = seg.match(/^(.+?)\((.*?)\)\s*[—\-]\s*(.+)$/)
        if (m) {
          const ja = m[1].trim()
          const read = m[2].trim()
          const en = m[3].trim()
          if (hasKanjiOrKatakana(ja)) {
            out.push({ ja, read: read || undefined, en, ctx })
          }
          continue
        }
        // Fallback: JA — EN
        const m2 = seg.match(/^(.+?)\s*[—\-]\s*(.+)$/)
        if (m2) {
          const ja = m2[1].trim()
          const en = m2[2].trim()
          if (hasKanjiOrKatakana(ja)) {
            out.push({ ja, en, ctx })
          }
        }
      }
    }
    return out
  }

  function dedupeVocabs(arr: Vocab[]): Array<Vocab & { count: number }> {
    const map = new Map<string, { ja: string, read?: string, en: string, count: number, idx: number }>()
    arr.forEach((v, idx) => {
      const key = (v.ja || '').trim()
      if (!key) return
      const ex = map.get(key)
      if (ex) {
        ex.count += 1
        if (!ex.read && v.read) ex.read = v.read
        if (!ex.en && v.en) ex.en = v.en
      } else {
        map.set(key, { ja: v.ja, read: v.read, en: v.en, count: 1, idx })
      }
    })
    return Array.from(map.values()).sort((a, b) => a.idx - b.idx)
  }

  function accrueCounts(arr: Vocab[]): Array<Vocab & { count: number }> {
    const counts = new Map<string, number>()
    const out: Array<Vocab & { count: number }> = []
    for (const v of arr) {
      const key = (v.ja || '').trim()
      if (!key) continue
      const n = (counts.get(key) || 0) + 1
      counts.set(key, n)
      out.push({ ...v, count: n })
    }
    return out
  }

  function buildCsv(): string {
    const rows = [["ja","reading","en","count"], ...vocabsUniq.map(v => [v.ja, v.read || '', v.en, String(v.count)])]
    return rows.map(r => r.map(x => '"' + String(x).replace(/"/g,'""') + '"').join(',')).join('\n')
  }

  function isoFileNameNow(): string {
    const d = new Date()
    const pad = (n: number, w=2) => String(n).padStart(w, '0')
    const ms = pad(d.getMilliseconds(), 3)
    const name = `vocab_report-${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}-${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}-${ms}.csv`
    return name
  }

  async function exportCsvPeriodic() {
    try {
      const csv = buildCsv()
      const fileName = isoFileNameNow()
      await invoke<string>('export_vocabs', { csv, fileName })
    } catch (e) {
      console.error('Periodic export failed', e)
    }
  }

  function rawTranscriptText(): string {
    try { return rawTimeline.map(r => r.text).join('\n') } catch { return '' }
  }

  async function exportRawTranscriptOnClose() {
    try {
      const text = rawTranscriptText()
      if (text && text.trim()) {
        const d = new Date()
        const pad = (n: number, w=2) => String(n).padStart(w, '0')
        const ms = pad(d.getMilliseconds(), 3)
        const name = `raw_transcript-${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}-${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}-${ms}.txt`
        await invoke<string>('export_raw_transcript', { text, fileName: name })
      }
    } catch (e) {
      console.error('Export raw transcript failed', e)
    }
  }

  // Hover handlers for vocabulary items
  function highlightContext(ctx?: string) {
    if (!ctx) return
    const container = transcriptEl?.querySelector('.content')
    if (!container) return
    if (highlightedEl) highlightedEl.classList.remove('vocab-highlight')
    highlightedEl = null
    const nodes = Array.from(container.querySelectorAll('p, li')) as HTMLElement[]
    for (const n of nodes) {
      if (n.textContent && n.textContent.includes(ctx)) {
        n.classList.add('vocab-highlight')
        highlightedEl = n
        break
      }
    }
  }
  function clearHighlight() {
    if (highlightedEl) highlightedEl.classList.remove('vocab-highlight')
    highlightedEl = null
  }
  function onVocabEnter(e: MouseEvent, v: Vocab) {
    const el = e.currentTarget as HTMLElement
    if (!el) return
    const r = el.getBoundingClientRect()
    hoverVocab = v
    hoverTop = Math.max(8, r.top)
    hoverLeft = r.right + 10
    hoverVisible = true
    highlightContext(v.ctx)
  }
  function onVocabMove(e: MouseEvent) {
    if (!hoverVisible) return
    hoverTop = Math.max(8, e.clientY - 16)
    const vw = window.innerWidth || document.documentElement.clientWidth
    const bubbleWidth = 480
    const rightPref = e.clientX + 12 + bubbleWidth
    if (rightPref < vw - 8) { hoverLeft = e.clientX + 12 } else { hoverLeft = Math.max(8, e.clientX - 12 - bubbleWidth) }
  }
  function onVocabLeave() {
    hoverVisible = false
    hoverVocab = null
    clearHighlight()
  }

  function buildSuggestions(text: string, exclude: Set<string>): Suggest[] {
    const lines = text.split(/\r?\n/).slice(-60)
    const out: Suggest[] = []
    const seen = new Set<string>()
    for (const raw of lines) {
      const line = stripTags(raw)
      if (!line || /^(Vocab:|📚\s*Context Vocab:)/.test(line)) continue
      const re = /[\u30A0-\u30FF]+|[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF々〆ヵヶ]+[\wぁ-ゖァ-ヺー]*/g
      const m = line.match(re)
      if (!m) continue
      for (const ja of m) {
        const key = ja.trim()
        if (!key || exclude.has(key) || seen.has(key)) continue
        seen.add(key)
        out.push({ ja: key, ctx: line })
        if (out.length >= 30) break
      }
      if (out.length >= 30) break
    }
    return out
  }
</script>

<div class="container">
  <SidebarPanel side="left" title="Vocabs" count={vocabsUniq.length} open={sidebarOpen} widthOpen={SIDEBAR_W_OPEN} widthClosed={SIDEBAR_W_CLOSED} fluid={true} on:toggle={() => sidebarOpen = !sidebarOpen}>
    <div class="vocabs-list" on:scroll={() => { hoverVisible = false; clearHighlight() }}>
      {#each vocabsTimeline as v}
        <div class="vocab-item"
             on:mouseenter={(e) => onVocabEnter(e, v)}
             on:mousemove={(e) => onVocabMove(e)}
             on:mouseleave={onVocabLeave}>
          <span class="vocab-ja">{v.ja}</span>
          {#if v.read}<span class="vocab-read">{v.read}</span>{/if}
          <span class="vocab-en">{v.en}</span>
          {#if v.count > 1}<span class="vocab-count">×{v.count}</span>{/if}
        </div>
      {/each}
    </div>
  </SidebarPanel>

  <div class="main">
    <div class="panel transcript" bind:this={transcriptEl} on:scroll={updateScrollState} on:mouseup={updateScrollState}>
      <div class="content" on:dblclick={scrollToBottom}>
        {#if asrBackend === 'openai'}
          {@html trustHtml(transcriptHtml)}
        {:else}
          {#each rawTimeline as r}
            <p>{r.text}</p>
          {/each}
        {/if}
      </div>
    </div>

    <div class="panel wave">
      <canvas bind:this={waveCanvas} on:resize={drawWave} on:mousedown={drawWave}></canvas>
    </div>

    {#if showScrollArrow}
      <div class="scroll-arrow" on:click={scrollToBottom}>▼ New</div>
    {/if}
  </div>

  {#if detailsEnabled}
    <SidebarPanel side="right" title="Suggestions" count={suggTimeline.length} open={rightOpen} widthOpen={RIGHT_W_OPEN} widthClosed={RIGHT_W_CLOSED} fluid={true} on:toggle={() => rightOpen = !rightOpen}>
      <div class="vocabs-list">
        {#each suggTimeline.filter(s => !vocabsUniq.find(v => v.ja === s.ja)) as s}
          <div class="vocab-item">
            <span class="vocab-ja">{s.ja}</span>
            {#if s.read}<span class="vocab-read">{s.read}</span>{/if}
            {#if s.en}<span class="vocab-en">{s.en}</span>{/if}
            {#if s.hint}<span class="vocab-hint">{s.hint}</span>{/if}
            {#if s.count>1}<span class="vocab-count">×{s.count}</span>{/if}
          </div>
        {/each}
      </div>
    </SidebarPanel>
  {/if}

  

  {#if asrDebugEnabled}
  <SidebarPanel side="debug" title="ASR Debug" count={asrRows.length} open={debugOpen} widthOpen={DEBUG_W_OPEN} widthClosed={DEBUG_W_CLOSED} on:toggle={() => debugOpen = !debugOpen}>
    <div class="vocabs-list">
      {#each asrRows as r}
        <div class="pair">
          <div class="asr-col">
            <div class="asr-title">OpenAI</div>
            <div class="asr-card">{r.openai?.text || ''}</div>
            <div class="asr-meta">{r.openai?.ms ? `${r.openai.ms} ms` : ''}</div>
          </div>
          <div class="asr-col">
            <div class="asr-title">Home</div>
            <div class="asr-card">{r.remote?.text || ''}</div>
            <div class="asr-meta">{r.remote?.ms ? `${r.remote.ms} ms` : ''}</div>
          </div>
        </div>
      {/each}
    </div>
  </SidebarPanel>
  {/if}
</div>

{#if hoverVisible && hoverVocab}
  <div class="hover-bubble" style={`top:${hoverTop}px; left:${hoverLeft}px;`}>
    <div class="hover-header">
      <span class="hover-ja">{hoverVocab.ja}</span>
      {#if hoverVocab.read}
        <span class="hover-reading">({hoverVocab.read})</span>
      {/if}
    </div>
    <div class="hover-meanings">
      {#if hoverVocab.en}
        <div class="meaning-item">• {hoverVocab.en}</div>
      {/if}
    </div>
    {#if hoverVocab.ctx}
      <div class="hover-section">
        <div class="section-title">Context:</div>
        <div class="context-text">{hoverVocab.ctx}</div>
      </div>
    {/if}
  </div>
{/if}
