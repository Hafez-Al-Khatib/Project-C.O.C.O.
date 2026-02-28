<script lang="ts">
  import { onMount } from 'svelte';
  import {
    BrainCircuit, Database, Cpu, Eye, MessageSquare,
    ArrowRight, CheckCircle2, AlertTriangle, Loader2,
    ChevronDown, RefreshCw, Sparkles, Clock, Target,
    Wrench
  } from 'lucide-svelte';

  interface TraceStep {
    type: 'thought' | 'action' | 'observation' | 'final_answer' | 'error';
    content?: string;
    tool?: string;
    input?: string;
  }

  let trace: TraceStep[] = [];
  let finalAnswer = '';
  let agentState: 'idle' | 'thinking' | 'complete' | 'error' = 'idle';
  let branchInput = 'Conut Jnah';
  let monthInput = 11;
  let yearInput = 2026;
  let queryInput = '';
  let elapsedMs = 0;

  $: queryInput = `Analyze ${branchInput} for ${new Date(yearInput, monthInput - 1).toLocaleString('default', { month: 'long' })} ${yearInput}: forecast demand, allocate staffing, evaluate expansion opportunities, and optimize menu combos.`;

  async function runAgent() {
    agentState = 'thinking';
    trace = [];
    finalAnswer = '';
    const t0 = performance.now();

    try {
      const res = await fetch('http://localhost:8000/openclaw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryInput,
          context: { branch_name: branchInput, month: monthInput, year: yearInput }
        })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      trace = data.trace || [];
      finalAnswer = data.final_answer || '';
      agentState = 'complete';
    } catch (e: any) {
      agentState = 'error';
      finalAnswer = `Agent failed: ${e.message}`;
      trace = [{ type: 'error', content: e.message }];
    }

    elapsedMs = Math.round(performance.now() - t0);
  }

  function getStepIcon(type: string) {
    switch(type) {
      case 'thought': return BrainCircuit;
      case 'action': return Wrench;
      case 'observation': return Eye;
      case 'final_answer': return Target;
      default: return AlertTriangle;
    }
  }

  function getStepLabel(type: string) {
    switch(type) {
      case 'thought': return 'Thought';
      case 'action': return 'Action';
      case 'observation': return 'Observation';
      case 'final_answer': return 'Final Answer';
      default: return 'Error';
    }
  }

  function getStepColor(type: string) {
    switch(type) {
      case 'thought': return { bg: 'bg-violet-50', border: 'border-violet-200', text: 'text-violet-700', badge: 'bg-violet-100 text-violet-700' };
      case 'action': return { bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-700', badge: 'bg-amber-100 text-amber-700' };
      case 'observation': return { bg: 'bg-sky-50', border: 'border-sky-200', text: 'text-sky-700', badge: 'bg-sky-100 text-sky-700' };
      case 'final_answer': return { bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-700', badge: 'bg-emerald-100 text-emerald-700' };
      default: return { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', badge: 'bg-red-100 text-red-700' };
    }
  }

  let expandedObs: {[key: number]: boolean} = {};
  function toggleObs(i: number) { expandedObs[i] = !expandedObs[i]; }
</script>

<svelte:head>
  <title>C.O.C.O. — Chief of Operations Copilot (ReAct Agent)</title>
</svelte:head>

<div class="space-y-6">

  <!-- Header -->
  <div class="pb-6 border-b border-slate-200">
    <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-50 border border-violet-200 text-violet-700 text-xs font-semibold uppercase tracking-wider mb-3">
      <BrainCircuit class="w-3.5 h-3.5" /> ReAct Agent · LangGraph
    </div>
    <h1 class="text-3xl font-extrabold tracking-tight text-slate-900">
      Chief of Operations Copilot
    </h1>
    <p class="mt-2 text-slate-500 max-w-2xl leading-relaxed text-sm">
      This agent follows the <strong>ReAct</strong> (Reasoning + Acting) framework powered by <strong>LangGraph</strong>. 
      It chains SQL queries, ML model inference, and strategic analysis into a single reasoning pass.
    </p>
  </div>

  <!-- Controls -->
  <div class="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-4">
    <div class="flex flex-wrap items-end gap-4">
      <div class="flex flex-col gap-1.5">
        <label class="text-xs font-semibold text-slate-500 uppercase tracking-wider" for="branch">Branch</label>
        <select id="branch" bind:value={branchInput} class="px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-500/30">
          <option>Conut Jnah</option>
          <option>Conut Main</option>
          <option>Conut Tyre</option>
          <option>Main Street Coffee</option>
        </select>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-xs font-semibold text-slate-500 uppercase tracking-wider" for="month">Month</label>
        <select id="month" bind:value={monthInput} class="px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-500/30">
          {#each Array.from({length: 12}, (_, i) => i + 1) as m}
            <option value={m}>{new Date(2026, m - 1).toLocaleString('default', { month: 'long' })}</option>
          {/each}
        </select>
      </div>
      <div class="flex flex-col gap-1.5">
        <label class="text-xs font-semibold text-slate-500 uppercase tracking-wider" for="year">Year</label>
        <input id="year" type="number" bind:value={yearInput} class="px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm font-medium text-slate-800 w-24 focus:outline-none focus:ring-2 focus:ring-violet-500/30" />
      </div>
      <button
        on:click={runAgent}
        disabled={agentState === 'thinking'}
        class="ml-auto px-5 py-2.5 rounded-xl font-semibold text-sm shadow-sm transition-all
          {agentState === 'thinking'
            ? 'bg-slate-200 text-slate-400 cursor-wait'
            : 'bg-gradient-to-r from-violet-600 to-indigo-600 text-white hover:shadow-md hover:shadow-violet-500/20 hover:scale-[1.02] active:scale-[0.98]'
          }"
      >
        {#if agentState === 'thinking'}
          <span class="flex items-center gap-2"><Loader2 class="w-4 h-4 animate-spin" /> Agent Reasoning...</span>
        {:else if agentState === 'complete'}
          <span class="flex items-center gap-2"><RefreshCw class="w-4 h-4" /> Re-run Agent</span>
        {:else}
          <span class="flex items-center gap-2"><Sparkles class="w-4 h-4" /> Run ReAct Agent</span>
        {/if}
      </button>
    </div>

    <!-- Query Preview -->
    <div class="bg-slate-50 rounded-xl px-4 py-3 border border-slate-100">
      <p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1.5">
        <MessageSquare class="w-3 h-3" /> Agent Query
      </p>
      <p class="text-sm text-slate-600">{queryInput}</p>
    </div>
  </div>

  <!-- Loading -->
  {#if agentState === 'thinking'}
    <div class="flex flex-col items-center justify-center py-16 space-y-4">
      <div class="relative">
        <div class="h-14 w-14 border-4 border-violet-100 border-t-violet-500 rounded-full animate-spin"></div>
        <BrainCircuit class="w-6 h-6 text-violet-500 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
      </div>
      <p class="text-slate-500 font-medium">Agent is executing ReAct loop...</p>
      <p class="text-xs text-slate-400">Chaining: SQL Engine → Model Inference → Strategy → Synthesis</p>
    </div>
  {/if}

  <!-- ReAct Trace -->
  {#if trace.length > 0 && agentState !== 'thinking'}
    <div class="space-y-0">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-lg font-bold text-slate-800 flex items-center gap-2">
          Reasoning Trace
          <span class="text-xs font-medium text-slate-400 bg-slate-100 px-2 py-0.5 rounded-full">{trace.length} steps</span>
        </h2>
        {#if elapsedMs > 0}
          <span class="text-xs text-slate-400 flex items-center gap-1">
            <Clock class="w-3 h-3" /> {elapsedMs}ms total
          </span>
        {/if}
      </div>

      {#each trace as step, i}
        {@const colors = getStepColor(step.type)}
        <div class="flex gap-3">
          <!-- Vertical line -->
          <div class="flex flex-col items-center">
            <div class="h-7 w-7 rounded-full {colors.bg} border {colors.border} flex items-center justify-center flex-shrink-0 z-10">
              <svelte:component this={getStepIcon(step.type)} class="w-3.5 h-3.5 {colors.text}" />
            </div>
            {#if i < trace.length - 1}
              <div class="w-px flex-1 bg-slate-200 min-h-[16px]"></div>
            {/if}
          </div>

          <!-- Content -->
          <div class="pb-4 flex-1 min-w-0">
            <div class="flex items-center gap-2 mb-1">
              <span class="text-xs font-bold {colors.badge} px-2 py-0.5 rounded-md uppercase tracking-wider">
                {getStepLabel(step.type)}
              </span>
              {#if step.tool}
                <span class="text-xs font-mono text-slate-400 bg-slate-100 px-2 py-0.5 rounded">
                  {step.tool}
                </span>
              {/if}
            </div>

            {#if step.type === 'thought'}
              <p class="text-sm text-slate-600 leading-relaxed">{step.content}</p>
            {:else if step.type === 'action'}
              <pre class="text-xs font-mono text-amber-800 bg-amber-50 border border-amber-100 rounded-lg px-3 py-2 overflow-x-auto whitespace-pre-wrap">{step.input || step.content}</pre>
            {:else if step.type === 'observation'}
              <div>
                <button on:click={() => toggleObs(i)} class="text-xs font-medium text-sky-600 hover:text-sky-800 transition-colors flex items-center gap-1 mb-1">
                  <ChevronDown class="w-3 h-3 transition-transform {expandedObs[i] ? 'rotate-180' : ''}" />
                  {expandedObs[i] ? 'Collapse' : 'Expand'} output ({(step.content || '').length} chars)
                </button>
                {#if expandedObs[i]}
                  <pre class="text-xs font-mono text-slate-700 bg-slate-900 text-emerald-400 rounded-lg px-3 py-2 overflow-x-auto max-h-48 whitespace-pre-wrap">{step.content}</pre>
                {/if}
              </div>
            {:else if step.type === 'final_answer'}
              <div class="bg-emerald-50 border border-emerald-200 rounded-xl px-4 py-3">
                <p class="text-sm text-slate-700 leading-relaxed whitespace-pre-line">{step.content}</p>
              </div>
            {:else}
              <p class="text-sm text-red-600">{step.content}</p>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {/if}

  <!-- Empty State -->
  {#if agentState === 'idle'}
    <div class="flex flex-col items-center justify-center py-20 text-center">
      <div class="h-16 w-16 rounded-2xl bg-gradient-to-br from-violet-100 to-indigo-100 flex items-center justify-center mb-4 shadow-sm">
        <BrainCircuit class="h-8 w-8 text-violet-600" />
      </div>
      <h2 class="text-lg font-bold text-slate-800 mb-2">Ready to Reason</h2>
      <p class="text-sm text-slate-500 max-w-md leading-relaxed">
        Select a branch and period, then click <strong>Run ReAct Agent</strong>. The copilot will chain
        <strong>SQL queries</strong>, <strong>ML inference</strong>, and <strong>strategic analysis</strong> 
        into a full executive reasoning pass using the ReAct framework.
      </p>
      <div class="flex items-center gap-3 mt-6 text-xs text-slate-400">
        <span class="flex items-center gap-1 px-2 py-1 bg-slate-100 rounded-lg"><Database class="w-3 h-3" /> DuckDB SQL</span>
        <ArrowRight class="w-3 h-3" />
        <span class="flex items-center gap-1 px-2 py-1 bg-slate-100 rounded-lg"><Cpu class="w-3 h-3" /> GPR · BayesianRidge</span>
        <ArrowRight class="w-3 h-3" />
        <span class="flex items-center gap-1 px-2 py-1 bg-slate-100 rounded-lg"><Target class="w-3 h-3" /> Executive Brief</span>
      </div>
    </div>
  {/if}

</div>
