<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { ArrowUp, Command, Loader2, Sparkles, Activity, BrainCircuit, Wrench, Eye, AlertTriangle, BookOpen } from 'lucide-svelte';
  import { marked } from 'marked';

  type TraceStep = {
    type: 'thought' | 'action' | 'observation' | 'final_answer' | 'error';
    content?: string;
    tool?: string;
    input?: string;
  };
  
  type Message = {
    role: 'user' | 'assistant';
    content: string;
  };

  type AnalysisGroup = {
    id: number;
    messageIndex: number;
    steps: TraceStep[];
  };

  let messages: Message[] = [];
  let analyses: AnalysisGroup[] = [];
  let agentState: 'idle' | 'thinking' | 'error' = 'idle';
  
  let apiKey = '';
  let userQuery = '';
  let chatContainer: HTMLElement;
  let sanitize = (html: string) => html;

  onMount(async () => {
    const DOMPurify = (await import('dompurify')).default;
    sanitize = DOMPurify.sanitize;

    const savedKey = localStorage.getItem('gemini_api_key');
    if (savedKey) apiKey = savedKey;
    
    messages = [{
      role: 'assistant', 
      content: 'I am the Chief of Operations Copilot. Describe your strategic goal or issue, and I will analyze our models and databases to assist you.'
    }];
  });

  async function scrollToBottom() {
    await tick();
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }

  async function submitQuery() {
    if (!userQuery.trim() || agentState === 'thinking') return;
    if (apiKey) localStorage.setItem('gemini_api_key', apiKey);

    messages = [...messages, { role: 'user', content: userQuery }];
    userQuery = '';
    
    messages = [...messages, { role: 'assistant', content: '' }];
    const assistantIndex = messages.length - 1;
    
    const analysisId = Date.now();
    analyses = [{ id: analysisId, messageIndex: assistantIndex, steps: [] }, ...analyses];

    agentState = 'thinking';
    scrollToBottom();

    try {
      const res = await fetch('http://localhost:8000/openclaw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messages.slice(0, -1),
          gemini_api_key: apiKey,
          context: {}
        })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      
      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error("No reader");
      
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (!dataStr) continue;
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.type === 'token') {
                messages[assistantIndex].content += parsed.content;
                scrollToBottom();
              } else if (parsed.type === 'trace') {
                analyses = analyses.map(a => 
                  a.id === analysisId ? { ...a, steps: [...a.steps, parsed.step] } : a
                );
              } else if (parsed.type === 'error') {
                messages[assistantIndex].content += `\n\n**Error:** ${parsed.content}`;
                agentState = 'error';
              }
            } catch (e) {
              console.warn("Parse error for SSE data:", dataStr, e);
            }
          }
        }
      }
      setTimeout(() => { if (agentState !== 'error') agentState = 'idle'; }, 300);
    } catch (e: any) {
      agentState = 'error';
      messages[assistantIndex].content += `\n\nAgent failed: ${e.message}`;
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuery();
    }
  }

  function getStepIcon(type: string) {
    switch(type) {
      case 'thought': return BrainCircuit;
      case 'action': return Wrench;
      case 'observation': return Eye;
      default: return AlertTriangle;
    }
  }
</script>

<svelte:head>
  <title>C.O.C.O. Copilot</title>
</svelte:head>

<div class="h-screen flex bg-white font-sans text-slate-900 overflow-hidden text-[13px]">
  
  <!-- Left Side: Chat UI -->
  <main class="flex-1 flex flex-col items-center relative h-full bg-[#FAFAFA]">
    <!-- Header -->
    <header class="w-full flex justify-between items-center px-6 py-4 border-b border-slate-200 bg-white/50 backdrop-blur-md absolute top-0 z-10 shrink-0">
      <div class="flex items-center gap-3">
        <div class="w-7 h-7 bg-black rounded shadow-sm flex items-center justify-center">
          <Command class="w-4 h-4 text-white" />
        </div>
        <div class="flex flex-col">
          <span class="font-semibold text-[13px] tracking-tight text-black">C.O.C.O.</span>
          <span class="text-[10px] text-slate-400 font-medium">Chief of Operations AI Copilot</span>
        </div>
      </div>
      <div>
        <input 
          type="password" bind:value={apiKey} placeholder="Gemini API Key..." 
          class="px-3 py-1.5 w-56 text-[12px] bg-white border border-slate-200 rounded-lg outline-none focus:border-slate-400 focus:ring-1 focus:ring-slate-400 shadow-sm" 
        />
      </div>
    </header>

    <!-- Chat Messages -->
    <div bind:this={chatContainer} class="w-full max-w-3xl flex-1 overflow-y-auto px-6 pt-28 pb-32">
      <div class="space-y-8 min-h-full flex flex-col justify-end">
        {#each messages as msg}
          {#if msg.role === 'user'}
            <div class="flex justify-end animate-in fade-in slide-in-from-bottom-2 duration-300">
              <div class="max-w-[75%] bg-[#F4F4F5] border border-slate-200/60 text-[#111111] px-4 py-3 rounded-2xl rounded-tr-sm shadow-sm leading-relaxed text-[14px] font-medium whitespace-pre-wrap">
                {msg.content}
              </div>
            </div>
          {:else}
            <div class="flex gap-4 group">
              <div class="w-7 h-7 bg-white border border-slate-200 rounded-md flex items-center justify-center shrink-0 shadow-sm mt-0.5">
                {#if agentState === 'thinking' && !msg.content && msg === messages[messages.length-1]}
                  <Loader2 class="w-3.5 h-3.5 text-slate-400 animate-spin" />
                {:else}
                  <Sparkles class="w-3.5 h-3.5 text-slate-700" />
                {/if}
              </div>
              <div class="flex-1 max-w-[85%] prose prose-sm prose-slate max-w-none prose-p:leading-relaxed prose-pre:bg-slate-50 prose-pre:border prose-pre:border-slate-200 prose-pre:rounded-lg prose-pre:text-slate-700 prose-headings:font-semibold">
                {#if !msg.content && agentState === 'thinking'}
                  <div class="flex items-center gap-1.5 h-6 text-slate-400">
                    <span class="w-1.5 h-1.5 rounded-full bg-slate-300 animate-pulse"></span>
                    <span class="w-1.5 h-1.5 rounded-full bg-slate-300 animate-pulse delay-75"></span>
                    <span class="w-1.5 h-1.5 rounded-full bg-slate-300 animate-pulse delay-150"></span>
                  </div>
                {:else}
                  {@html sanitize(marked.parse(msg.content) || '')}
                {/if}
              </div>
            </div>
          {/if}
        {/each}
      </div>
    </div>

    <!-- Input Area -->
    <div class="absolute bottom-0 w-full max-w-3xl p-6 bg-gradient-to-t from-[#FAFAFA] via-[#FAFAFA] to-transparent shrink-0">
      <div class="relative bg-white border border-slate-200 rounded-xl shadow-sm focus-within:border-slate-300 focus-within:ring-4 focus-within:ring-slate-100 transition-all">
        <textarea 
          bind:value={userQuery} 
          on:keydown={handleKeydown}
          placeholder="Outline your strategic request..."
          class="w-full bg-transparent p-4 pr-14 outline-none resize-none min-h-[60px] text-[14px] placeholder-slate-400 rounded-xl"
        ></textarea>
        <button 
          on:click={submitQuery}
          disabled={agentState === 'thinking' || !userQuery.trim() || !apiKey}
          class="absolute right-2 bottom-2 w-8 h-8 flex items-center justify-center rounded-lg transition-colors
            {agentState === 'thinking' || !userQuery.trim() || !apiKey 
              ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
              : 'bg-black text-white hover:bg-zinc-800 shadow-sm'}"
        >
          <ArrowUp class="w-4 h-4" />
        </button>
      </div>
      {#if !apiKey}
        <div class="absolute -top-2 left-1/2 -translate-x-1/2 bg-red-50 text-red-600 px-3 py-1 rounded-full text-[10px] font-bold border border-red-100 shadow-sm flex items-center gap-1">
          <AlertTriangle class="w-3 h-3" /> API Key Required
        </div>
      {/if}
    </div>
  </main>

  <!-- Right Side: Knowledge Bank / Inspector -->
  <aside class="w-[420px] border-l border-slate-200 bg-white flex flex-col h-full shrink-0 shadow-[-4px_0_24px_rgba(0,0,0,0.02)] z-20">
    <header class="px-5 py-4 border-b border-slate-100 flex justify-between items-center bg-white shrink-0">
      <div class="flex items-center gap-2 text-[13px] font-semibold text-slate-800">
        <Activity class="w-4 h-4 text-slate-400" />
        Inspector
      </div>
      <div class="text-[10px] font-bold px-2 py-0.5 bg-slate-100 text-slate-500 rounded uppercase tracking-widest border border-slate-200/60">
        {analyses.filter(a => a.steps.length > 0).length} Tickers
      </div>
    </header>

    <div class="flex-1 overflow-y-auto p-4 space-y-4">
      {#if analyses.filter(a => a.steps.length > 0).length === 0}
         <div class="h-full flex flex-col items-center justify-center text-slate-400 space-y-3 opacity-80">
           <div class="w-12 h-12 bg-slate-50 border border-slate-200 border-dashed rounded-xl flex items-center justify-center">
             <BookOpen class="w-5 h-5 text-slate-300" />
           </div>
           <p class="text-[12px] text-center max-w-[200px] leading-relaxed">Agent reasoning and backend tool execution will appear here.</p>
         </div>
      {/if}

      {#each analyses as analysis}
        {#if analysis.steps.length > 0}
          <div class="bg-white border border-slate-200 rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.02)] overflow-hidden animate-in slide-in-from-right-4 fade-in duration-300">
            <div class="bg-slate-50/80 px-4 py-2.5 border-b border-slate-100 flex justify-between items-center">
              <span class="font-semibold text-[11px] text-slate-600 uppercase tracking-widest">Job #{analysis.id.toString().slice(-4)}</span>
              <span class="text-[10px] text-slate-400 font-medium">Turn {analysis.messageIndex}</span>
            </div>
            
            <div class="divide-y divide-slate-100">
              {#each analysis.steps as step}
                <div class="px-4 py-3 flex items-start gap-3 hover:bg-slate-50/50 transition-colors">
                  <div class="mt-0.5 text-slate-400">
                    <svelte:component this={getStepIcon(step.type)} class="w-3.5 h-3.5" />
                  </div>
                  <div class="min-w-0 flex-1">
                    <span class="font-bold text-[9px] uppercase tracking-widest text-slate-400 mb-1 block">{step.type}</span>
                    
                    {#if step.type === 'action'}
                      <div class="font-mono text-[10px] bg-slate-50 p-2 rounded-md text-slate-700 break-all border border-slate-200 shadow-sm inline-block">
                        <span class="font-bold text-slate-900">{step.tool}</span><span class="text-slate-500">({step.input})</span>
                      </div>
                    {:else if step.type === 'observation'}
                      <div class="text-slate-600 font-mono text-[10px] bg-white border border-slate-200 p-2 rounded-md max-h-32 overflow-y-auto shadow-sm whitespace-pre-wrap">
                        {step.content}
                      </div>
                    {:else}
                      <div class="text-slate-700 leading-relaxed text-[12px]">
                        {step.content}
                      </div>
                    {/if}
                  </div>
                </div>
              {/each}
            </div>
          </div>
        {/if}
      {/each}
    </div>
  </aside>
</div>
