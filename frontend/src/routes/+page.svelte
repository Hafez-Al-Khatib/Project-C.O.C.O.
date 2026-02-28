<script lang="ts">
  import { onMount } from 'svelte';
  import {
    Activity,
    Network,
    MapPin,
    ArrowUpRight,
    Users,
    TrendingUp,
    BarChart3,
    Coffee,
    RefreshCw,
    CheckCircle2,
    XCircle
  } from 'lucide-svelte';

  // State
  let loading = true;
  let backendStatus = 'checking'; // 'checking', 'online', 'error'
  
  // Real Data State
  let demandPrediction: any = null;
  let staffingEstimation: any = null;
  let expansionScore: any = null;
  let comboOptimizations: any = null;

  async function fetchBackendData() {
    loading = true;
    try {
      // 1. Check Demand Forecaster (Obj 2)
      const demandRes = await fetch('http://localhost:8000/tools/predict_demand', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ branch_name: 'Conut Jnah', month: 11, year: 2026 })
      });
      demandPrediction = await demandRes.json();

      // 2. Check Staffing (Obj 4) 
      const staffRes = await fetch('http://localhost:8000/tools/estimate_staffing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ branch_name: 'Conut Jnah', predicted_volume: demandPrediction?.predicted_volume || 1500 })
      });
      staffingEstimation = await staffRes.json();

      // 3. Check Expansion (Obj 3)
      const expRes = await fetch('http://localhost:8000/tools/expansion_feasibility', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          candidate_lat: 34.4346, 
          candidate_lon: 35.8362,
          candidate_features: {
            coffee_ratio: 0.45,
            pastry_ratio: 0.35,
            drinks_ratio: 0.15,
            shakes_ratio: 0.05
          }
        })
      });
      expansionScore = await expRes.json();

      // 4. Check Combos (Obj 1)
      const comboRes = await fetch('http://localhost:8000/tools/get_combos', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_item: 'Cafe Latte', top_n: 3 })
      });
      comboOptimizations = await comboRes.json();

      backendStatus = 'online';
    } catch (e) {
      console.error("Backend connection failed", e);
      backendStatus = 'error';
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    fetchBackendData();
  });

</script>

<svelte:head>
  <title>Project C.O.C.O. - Analytics Control Center</title>
</svelte:head>

<div class="space-y-10 animate-in fade-in zoom-in-95 duration-700">
  
  <!-- Premium Header -->
  <div class="flex flex-col md:flex-row gap-6 md:items-end justify-between border-b border-zinc-200 pb-8">
    <div>
      <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-700 text-xs font-semibold uppercase tracking-wider mb-4">
        <Activity class="w-3.5 h-3.5" /> Live Production Mode
      </div>
      <h1 class="text-4xl font-extrabold tracking-tight text-zinc-900">
        AI Control Center
      </h1>
      <p class="mt-3 text-lg text-zinc-500 max-w-2xl leading-relaxed">
        Centralized manifest for the Conut Operational Copilot. Real-time inference across all core business intelligence models.
      </p>
    </div>
    
    <div class="flex items-center gap-4 bg-white rounded-2xl border border-zinc-200 shadow-sm px-5 py-3">
      <div class="flex flex-col">
        <span class="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Gateway Status</span>
        <div class="flex items-center gap-2 mt-1">
          {#if backendStatus === 'online'}
            <div class="h-2.5 w-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)] animate-pulse"></div>
            <span class="text-sm font-bold text-zinc-700">Online & Syncing</span>
          {:else if backendStatus === 'error'}
            <div class="h-2.5 w-2.5 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.4)]"></div>
            <span class="text-sm font-bold text-zinc-700">Connection Failed</span>
          {:else}
            <RefreshCw class="h-3 w-3 text-amber-500 animate-spin" />
            <span class="text-sm font-bold text-zinc-700">Polling...</span>
          {/if}
        </div>
      </div>
      <button on:click={fetchBackendData} class="p-2 hover:bg-zinc-100 rounded-lg transition-colors text-zinc-400 hover:text-zinc-900">
        <RefreshCw class="w-4 h-4 {loading ? 'animate-spin' : ''}" />
      </button>
    </div>
  </div>

  {#if loading}
    <div class="flex flex-col items-center justify-center py-20 space-y-4">
      <div class="h-10 w-10 border-4 border-amber-200 border-t-amber-500 rounded-full animate-spin"></div>
      <p class="text-zinc-500 font-medium animate-pulse">Running live MLOps inference routines...</p>
    </div>
  {:else}
    <!-- Live Data Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
      
      <!-- Objective 2: Demand -->
      <div class="bg-white rounded-3xl border border-zinc-200 shadow-sm overflow-hidden hover:shadow-md transition-shadow">
        <div class="p-6 border-b border-zinc-100 bg-gradient-to-r from-orange-50 to-white flex justify-between items-start">
          <div>
            <div class="flex items-center gap-2 mb-2">
              <span class="px-2 py-0.5 rounded-md bg-orange-100 text-orange-700 text-xs font-bold">Objective 2</span>
              <span class="text-xs text-zinc-500 font-medium">{demandPrediction?.model_type || 'Unknown Model'}</span>
            </div>
            <h3 class="text-xl font-bold text-zinc-900 flex items-center gap-2">
              <TrendingUp class="w-5 h-5 text-orange-500" /> Demand Forecaster
            </h3>
          </div>
          <div class="bg-white p-2 rounded-xl shadow-sm border border-zinc-100">
            <span class="text-xs font-bold text-zinc-400 uppercase block mb-1">Target Branch</span>
            <span class="text-sm font-semibold text-zinc-800">{demandPrediction?.branch || 'N/A'}</span>
          </div>
        </div>
        <div class="p-6">
          <div class="flex items-end gap-3 mb-6">
            <span class="text-4xl font-extrabold text-zinc-900">{(demandPrediction?.predicted_volume || 0).toLocaleString()}</span>
            <span class="text-zinc-500 font-medium mb-1">Est. Transactions</span>
          </div>
          <div class="space-y-3">
            <div class="flex justify-between items-center text-sm py-2 border-b border-zinc-100">
              <span class="text-zinc-500">Confidence Interval</span>
              <span class="font-semibold text-zinc-800">{demandPrediction?.confidence_interval || 'N/A'}</span>
            </div>
            <div class="flex justify-between items-center text-sm py-2 border-b border-zinc-100">
              <span class="text-zinc-500">Model MAPE (Error)</span>
              <span class="font-semibold px-2 py-1 rounded bg-green-50 text-green-700">{demandPrediction?.mape?.toFixed(1) || '0'}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Objective 4: Staffing -->
      <div class="bg-white rounded-3xl border border-zinc-200 shadow-sm overflow-hidden hover:shadow-md transition-shadow">
        <div class="p-6 border-b border-zinc-100 bg-gradient-to-r from-blue-50 to-white flex justify-between items-start">
          <div>
             <div class="flex items-center gap-2 mb-2">
              <span class="px-2 py-0.5 rounded-md bg-blue-100 text-blue-700 text-xs font-bold">Objective 4</span>
              <span class="text-xs text-zinc-500 font-medium">{staffingEstimation?.model_type || 'Unknown Model'}</span>
            </div>
            <h3 class="text-xl font-bold text-zinc-900 flex items-center gap-2">
              <Users class="w-5 h-5 text-blue-500" /> Shift Staffing Allocation
            </h3>
          </div>
        </div>
        <div class="p-6">
          <div class="flex items-end gap-3 mb-6">
            <span class="text-4xl font-extrabold text-blue-600">{staffingEstimation?.recommended_staff || 0}</span>
            <span class="text-zinc-500 font-medium mb-1">Crew Members Req.</span>
          </div>
          <div class="bg-zinc-50 rounded-xl p-4 border border-zinc-100">
            <h4 class="text-xs font-bold text-zinc-500 uppercase mb-3">XAI Inference Drivers</h4>
            <div class="grid grid-cols-2 gap-4">
               {#if staffingEstimation?.xai_drivers}
                 {#each Object.entries(staffingEstimation.xai_drivers) as [key, value]}
                   <div>
                     <span class="block text-xs text-zinc-500 truncate" title={key}>{key.replace(/_/g, ' ')}</span>
                     <span class="block text-sm font-semibold text-zinc-800 truncate" title={String(value)}>{value}</span>
                   </div>
                 {/each}
               {:else}
                 <span class="text-sm text-zinc-400">No driver data</span>
               {/if}
            </div>
          </div>
        </div>
      </div>

      <!-- Objective 3: Expansion -->
      <div class="bg-white rounded-3xl border border-zinc-200 shadow-sm overflow-hidden hover:shadow-md transition-shadow">
        <div class="p-6 border-b border-zinc-100 bg-gradient-to-r from-emerald-50 to-white flex justify-between items-start">
          <div>
            <div class="flex items-center gap-2 mb-2">
              <span class="px-2 py-0.5 rounded-md bg-emerald-100 text-emerald-700 text-xs font-bold">Objective 3</span>
              <span class="text-xs text-zinc-500 font-medium">OSM Live Integration</span>
            </div>
            <h3 class="text-xl font-bold text-zinc-900 flex items-center gap-2">
              <MapPin class="w-5 h-5 text-emerald-500" /> Real Estate Expansion
            </h3>
          </div>
        </div>
        <div class="p-6">
           <div class="flex items-center gap-4 mb-6">
            <div class="flex-1">
              <span class="block text-xs font-medium text-zinc-500 mb-1">Target Score (vs {expansionScore?.reference_branch || 'Top Branch'})</span>
              <div class="flex items-end gap-2">
                <span class="text-3xl font-extrabold text-emerald-600">{(expansionScore?.similarity_score || 0).toFixed(2)}</span>
                <span class="text-sm text-zinc-400 mb-1">/ 1.00</span>
              </div>
            </div>
            <div class="h-12 w-px bg-zinc-200"></div>
            <div class="flex-1">
              <span class="block text-xs font-medium text-zinc-500 mb-1">Recommendation</span>
              <span class="text-sm font-semibold text-zinc-800 line-clamp-2">{expansionScore?.recommendation || 'Pending evaluation'}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Objective 1: Combos -->
      <div class="bg-white rounded-3xl border border-zinc-200 shadow-sm overflow-hidden hover:shadow-md transition-shadow">
        <div class="p-6 border-b border-zinc-100 bg-gradient-to-r from-purple-50 to-white flex justify-between items-start">
          <div>
            <div class="flex items-center gap-2 mb-2">
              <span class="px-2 py-0.5 rounded-md bg-purple-100 text-purple-700 text-xs font-bold">Objective 1</span>
              <span class="text-xs text-zinc-500 font-medium">Louvain NetworkX</span>
            </div>
            <h3 class="text-xl font-bold text-zinc-900 flex items-center gap-2">
              <Network class="w-5 h-5 text-purple-500" /> Combo Optimizer
            </h3>
          </div>
        </div>
        <div class="p-6">
           <div class="flex items-center gap-3 mb-6">
             <span class="text-sm text-zinc-500 font-medium">Target Context:</span>
             <span class="px-3 py-1 bg-zinc-100 rounded-lg text-sm font-bold text-zinc-800">{comboOptimizations?.target_item || 'None'}</span>
             <ArrowUpRight class="w-4 h-4 text-zinc-400" />
             <span class="px-3 py-1 bg-purple-100 text-purple-800 rounded-lg text-lg font-bold shadow-sm">{comboOptimizations?.recommended_combo || 'None'}</span>
           </div>
           <p class="text-sm text-zinc-600 bg-zinc-50 p-4 rounded-xl border border-zinc-100">
             <span class="font-bold text-zinc-800 block mb-1">AI Reasoning:</span>
             {comboOptimizations?.business_reason || 'Waiting for graph traversal.'}
           </p>
        </div>
      </div>

    </div>
  {/if}
</div>
