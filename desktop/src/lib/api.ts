import { fetch as httpFetch } from '@tauri-apps/api/http'
import { invoke } from '@tauri-apps/api/tauri'

function nowMs(): number { try { return performance.now() } catch { return Date.now() } }
const LOG_ON = (() => {
  try {
    const v = String((import.meta as any)?.env?.VITE_TAURI_CLIENT_LOG ?? '').toLowerCase()
    if (v) return !(v === '0' || v === 'false' || v === 'off')
  } catch {}
  return true
})()
async function logClient(msg: string) { if (!LOG_ON) return; try { await invoke('client_log', { msg }) } catch {} }

export class ApiClient {
  baseUrl: string
  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, '')
  }

  async health() {
    const url = `${this.baseUrl}/api/health`
    const t0 = nowMs(); await logClient(`Api.health start url=${url}`)
    const r = await httpFetch(url, { method: 'GET' }).catch(async (e) => { await logClient(`Api.health error ${e?.message||e}`); throw e })
    await logClient(`Api.health status=${r.status} dur=${(nowMs()-t0).toFixed(0)}ms`)
    return r.data as any
  }

  async overlayTranscript() {
    const url = `${this.baseUrl}/api/overlay/transcript`
    const t0 = nowMs();
    const r = await httpFetch(url, { method: 'GET' }).catch(async (e) => { await logClient(`Api.overlayTranscript error ${e?.message||e}`); throw e })
    await logClient(`Api.overlayTranscript status=${r.status} dur=${(nowMs()-t0).toFixed(0)}ms`)
    if (r.status >= 400) throw new Error(`Transcript failed ${r.status}`)
    return r.data as { text?: string, html?: string }
  }

  async overlayWaveform() {
    const url = `${this.baseUrl}/api/overlay/waveform`
    const t0 = nowMs();
    const r = await httpFetch(url, { method: 'GET' }).catch(async (e) => { await logClient(`Api.overlayWaveform error ${e?.message||e}`); throw e })
    await logClient(`Api.overlayWaveform status=${r.status} dur=${(nowMs()-t0).toFixed(0)}ms`)
    if (r.status >= 400) throw new Error(`Waveform failed ${r.status}`)
    return r.data as { data: number[] }
  }

  async suggestions() {
    const url = `${this.baseUrl}/api/overlay/suggestions`
    const t0 = nowMs(); await logClient(`Api.suggestions start url=${url}`)
    const r = await httpFetch(url, { method: 'GET' }).catch(async (e) => { await logClient(`Api.suggestions error ${e?.message||e}`); throw e })
    await logClient(`Api.suggestions status=${r.status} dur=${(nowMs()-t0).toFixed(0)}ms`)
    if (r.status >= 400) throw new Error(`Suggestions failed ${r.status}`)
    return r.data as { items: Array<{ ja: string, read?: string, en?: string }> }
  }

  async asrDebugLog() {
    const url = `${this.baseUrl}/api/debug/asr`
    const t0 = nowMs(); await logClient(`Api.asrDebugLog start url=${url}`)
    const r = await httpFetch(url, { method: 'GET' }).catch(async (e) => { await logClient(`Api.asrDebugLog error ${e?.message||e}`); throw e })
    await logClient(`Api.asrDebugLog status=${r.status} dur=${(nowMs()-t0).toFixed(0)}ms`)
    if (r.status >= 400) throw new Error(`ASR debug failed ${r.status}`)
    return r.data as { items: Array<{ id: string, ts: number, openai?: { text?: string, ms?: number }, remote?: { text?: string, ms?: number } }> }
  }

  async enhancedVocab() {
    const url = `${this.baseUrl}/api/vocab/enhanced`
    const t0 = nowMs(); await logClient(`Api.enhancedVocab start url=${url}`)
    const r = await httpFetch(url, { method: 'GET' }).catch(async (e) => { await logClient(`Api.enhancedVocab error ${e?.message||e}`); throw e })
    await logClient(`Api.enhancedVocab status=${r.status} dur=${(nowMs()-t0).toFixed(0)}ms`)
    if (r.status >= 400) throw new Error(`Enhanced vocab failed ${r.status}`)
    return r.data as {
      enabled: boolean
      vocabulary: Array<{
        surface: string
        reading_hiragana: string
        reading_katakana?: string
        meaning_en: string
        word_type: string
        kanji_breakdown?: Array<{kanji: string, reading: string, meaning: string}>
        usage_notes?: string
        examples?: string[]
        line_index: number
      }>
      kanji_only: Array<{
        kanji: string
        readings_on: string[]
        readings_kun: string[]
        meaning_en: string
        stroke_count?: number
        line_index: number
      }>
      katakana_words: Array<{
        surface: string
        reading_hiragana: string
        meaning_en: string
        origin?: string
        line_index: number
      }>
      error?: string
    }
  }
}
