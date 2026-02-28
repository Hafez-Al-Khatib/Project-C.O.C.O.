<script lang="ts">
  import {
    Activity,
    Network,
    MapPin,
    ArrowUpRight,
    Users,
    TrendingUp,
    BarChart3,
    Coffee
  } from 'lucide-svelte';

  // Demo data reflecting our trained models
  const apiStatus = [
    { name: "Expansion Scorer", status: "Online", latency: "24ms", icon: MapPin },
    { name: "Staffing Estimator", status: "Online", latency: "18ms", icon: Users },
    { name: "Demand Forecaster", status: "Online", latency: "35ms", icon: TrendingUp },
    { name: "Combo Optimizer", status: "Online", latency: "12ms", icon: Network },
  ];

  const features = [
    {
      title: "Real Estate Expansion",
      description: "Scores candidate locations using dynamic OSM foot-traffic proxy metrics against best-performing branches.",
      status: "Verified (Obj 3)",
      color: "from-blue-500 to-cyan-400",
      icon: MapPin,
    },
    {
      title: "Demand Forecasting",
      description: "Combats holiday drift using Ratio-Based Target Engineering and walk-forward cross-validation on Prophet/GPR.",
      status: "Verified (Obj 2)",
      color: "from-amber-400 to-orange-500",
      icon: TrendingUp,
    },
    {
      title: "Combo Network Graph",
      description: "Identifies top co-purchase associations via Louvain Community Detection to drive bundle recommendations.",
      status: "Verified (Obj 1)",
      color: "from-purple-500 to-fuchsia-400",
      icon: Network,
    },
    {
      title: "Shift Staffing Allocation",
      description: "Maps predicted demand volume to optimal headcount using a dynamically tuned LightGBM regressor.",
      status: "Verified (Obj 4)",
      color: "from-emerald-400 to-teal-500",
      icon: Users,
    }
  ];
</script>

<svelte:head>
  <title>Project C.O.C.O. Dashboard</title>
</svelte:head>

<div class="space-y-8 animate-in fade-in duration-700">
  
  <!-- Header -->
  <div class="flex flex-col md:flex-row gap-6 md:items-end justify-between">
    <div>
      <h1 class="text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-zinc-100 to-zinc-500">
        AI Control Center
      </h1>
      <p class="mt-2 text-zinc-400 max-w-2xl px-1">
        Centralized manifest for the Conut Operational Copilot. OpenClaw connects to these backend APIs to automate decision-making across all 4 competitive tracks.
      </p>
    </div>
    <div class="flex items-center gap-3 rounded-full border border-zinc-800 bg-zinc-900/50 px-4 py-2 shadow-inner">
      <div class="h-2.5 w-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)] animate-pulse"></div>
      <span class="text-sm font-medium text-zinc-300">FastAPI Gateway Connected</span>
    </div>
  </div>

  <!-- Gateway Status Cards -->
  <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
    {#each apiStatus as api}
      <div class="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 transition-all hover:border-zinc-700 hover:bg-zinc-800/50 group">
        <div class="flex items-start justify-between">
          <div class="p-2 rounded-lg bg-zinc-800 group-hover:bg-zinc-700 transition-colors">
            <svelte:component this={api.icon} class="h-5 w-5 text-zinc-300" />
          </div>
          <span class="text-xs font-mono text-zinc-500">{api.latency}</span>
        </div>
        <div class="mt-4">
          <h3 class="text-sm font-semibold tracking-tight text-zinc-200">{api.name}</h3>
          <div class="mt-1 flex items-center gap-2">
            <div class="h-1.5 w-1.5 rounded-full bg-emerald-500"></div>
            <span class="text-xs font-medium text-zinc-400">{api.status}</span>
          </div>
        </div>
      </div>
    {/each}
  </div>

  <!-- Main Modules Grid -->
  <h2 class="text-xl font-bold tracking-tight text-zinc-100 pt-6">Active Capabilities</h2>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
    {#each features as feature}
      <div class="relative overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 p-6 shadow-xl transition-all hover:-translate-y-1 hover:border-zinc-700 group">
        <!-- Background Gradient Aura -->
        <div class="absolute -right-20 -top-20 h-48 w-48 rounded-full bg-gradient-to-br opacity-5 blur-3xl transition-opacity group-hover:opacity-10 {feature.color}"></div>
        
        <div class="relative z-10">
          <div class="flex items-center justify-between">
            <div class={`flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br ${feature.color} shadow-lg ring-1 ring-white/10`}>
              <svelte:component this={feature.icon} class="h-6 w-6 text-white drop-shadow-sm" />
            </div>
            <span class="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-1 text-xs font-medium text-emerald-400">
              <Activity class="h-3 w-3" />
              {feature.status}
            </span>
          </div>
          
          <div class="mt-6">
            <h3 class="text-lg font-semibold text-zinc-100 group-hover:text-amber-400 transition-colors">{feature.title}</h3>
            <p class="mt-2 text-sm leading-relaxed text-zinc-400">
              {feature.description}
            </p>
          </div>

          <div class="mt-6 flex items-center gap-2 text-sm font-medium text-zinc-500 hover:text-zinc-300 cursor-pointer transition-colors max-w-max">
            Explore Documentation
            <ArrowUpRight class="h-4 w-4" />
          </div>
        </div>
      </div>
    {/each}
  </div>
</div>
