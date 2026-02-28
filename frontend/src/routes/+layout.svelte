<script lang="ts">
  import '../app.css';
  import { Globe, Lightbulb, Users, BarChart3, Coffee } from 'lucide-svelte';
</script>

<div class="min-h-screen bg-zinc-950 text-zinc-100 selection:bg-amber-500/30">
  <nav class="sticky top-0 z-50 w-full border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-xl">
    <div class="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
      <div class="flex items-center gap-3">
        <div class="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-amber-400 to-orange-600 shadow-lg shadow-orange-500/20">
          <Coffee class="h-5 w-5 text-white" />
        </div>
        <span class="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-zinc-100 to-zinc-400">Project C.O.C.O.</span>
      </div>
      <div class="flex gap-6 text-sm font-medium text-zinc-400">
        <a href="/" class="hover:text-amber-400 transition-colors flex items-center gap-2">
          <Globe class="h-4 w-4" /> Expansion
        </a>
        <a href="/" class="hover:text-amber-400 transition-colors flex items-center gap-2">
          <Lightbulb class="h-4 w-4" /> Combos
        </a>
        <a href="/" class="hover:text-amber-400 transition-colors flex items-center gap-2">
          <BarChart3 class="h-4 w-4" /> Demand
        </a>
        <a href="/" class="hover:text-amber-400 transition-colors flex items-center gap-2">
          <Users class="h-4 w-4" /> Staffing
        </a>
      </div>
    </div>
  </nav>

  <main class="mx-auto max-w-7xl px-6 py-8">
    <slot />
  </main>
</div>
