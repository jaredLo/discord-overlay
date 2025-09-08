<script lang="ts">
  import { createEventDispatcher } from 'svelte'

  export let side: 'left' | 'right' | 'debug' = 'left'
  export let title = ''
  export let count: number | string = ''
  export let open = true
  export let widthOpen = 260
  export let widthClosed = 26
  export let fluid: boolean = false

  const dispatch = createEventDispatcher()
  function toggle() { dispatch('toggle') }

  $: sideClass = side === 'left' ? 'sidebar-left' : (side === 'right' ? 'sidebar-right' : 'sidebar-debug')
  $: styleWidth = fluid ? '' : `width:${open ? widthOpen : widthClosed}px`
  $: contentStyle = `opacity:${open ? 1 : 0}; pointer-events:${open ? 'auto' : 'none'}; transition: opacity 120ms ease;`
</script>

<div class={`sidebar ${sideClass}`} style={styleWidth}>
  <button type="button" class="sidebar-toggle" aria-label={open ? 'Collapse' : 'Expand'} on:click={toggle}>
    {#if open}{side !== 'left' ? '▶' : '◀'}{:else}{side !== 'left' ? '◀' : '▶'}{/if}
  </button>
  <div class="sidebar-content" style={contentStyle}>
    <div class="vocabs-header"><div class="sidebar-title">{title}</div><div><small style="margin-right:8px;">{count}</small></div></div>
    <div class="vocabs-list">
      <slot />
    </div>
  </div>
</div>


