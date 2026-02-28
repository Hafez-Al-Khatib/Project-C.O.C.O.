<script lang="ts">
  import '../app.css';
  import { Coffee, BrainCircuit } from 'lucide-svelte';
</script>

<div class="min-h-screen bg-gradient-to-b from-slate-50 via-white to-slate-50 text-slate-800 selection:bg-amber-500/30">
  <nav class="sticky top-0 z-50 w-full border-b border-slate-200/60 bg-white/70 backdrop-blur-xl">
    <div class="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
      <div class="flex items-center gap-3">
        <div class="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-amber-400 to-orange-600 shadow-lg shadow-orange-500/20">
          <Coffee class="h-5 w-5 text-white" />
        </div>
        <div class="flex flex-col">
          <span class="text-base font-bold tracking-tight text-slate-800">Project C.O.C.O.</span>
          <span class="text-[10px] text-slate-400 font-medium tracking-wider uppercase">Chief of Operations Copilot</span>
        </div>
      </div>
      <div class="flex items-center gap-2 bg-slate-100 px-3 py-1.5 rounded-full">
        <BrainCircuit class="h-4 w-4 text-amber-600" />
        <span class="text-xs font-semibold text-slate-600">Agent Mode</span>
      </div>
    </div>
  </nav>

  <main class="mx-auto max-w-6xl px-6 py-8">
    <slot />
  </main>

  <footer class="border-t border-slate-200/60 py-6">
    <div class="mx-auto max-w-6xl px-6 flex items-center justify-between text-xs text-slate-400">
      <span>Conut AI Engineering Hackathon 2026</span>
      <span>GPR · BayesianRidge · Louvain · OSM · MLFlow</span>
    </div>
  </footer>
</div>
