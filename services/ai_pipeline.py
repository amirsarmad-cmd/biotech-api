"""Fastest AI pipeline: 3 models FULLY PARALLEL, return immediately. Optional consensus pass."""
import os, re, logging, time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
logger = logging.getLogger(__name__)

def _extract_probability(text):
    # More lenient - handles bold markdown, ranges, approximations
    patterns = [
        # Bold markdown: **FINAL PROBABILITY: 72%**, **PROBABILITY: 72%**
        r'\*\*\s*(?:FINAL|CONSENSUS|REVISED|YOUR)?\s*PROBABILITY[:\s=]*(\d{1,3})\s*%?\s*\*\*',
        # Plain: FINAL PROBABILITY: 72%
        r'(?:final|consensus|revised|adjusted|triangulated|estimated|your)\s+probability[^\d\n]{0,20}(\d{1,3})\s*%',
        # Generic: probability ... 72%
        r'probability[^\d\n]{0,30}(\d{1,3})\s*%',
        # All caps header
        r'PROBABILITY[:\s=]+(\d{1,3})',
        # "72% probability" / "72% chance"
        r'(\d{1,3})\s*%\s*(?:probability|likelihood|chance|approval)',
        # "I estimate 72%", "my estimate is 72%"
        r'(?:estimate|assess|rate|put\s+(?:this|it)\s+at)[^\d\n]{0,15}(\d{1,3})\s*%',
        # Range like "70-75%" - take midpoint
        r'(\d{1,3})[-–](\d{1,3})\s*%',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                # Range pattern has 2 groups — take midpoint
                if m.lastindex and m.lastindex >= 2:
                    low = int(m.group(1))
                    high = int(m.group(2))
                    val = (low + high) // 2
                else:
                    val = int(m.group(1))
                if 0 <= val <= 100:
                    return val / 100.0
            except (ValueError, IndexError):
                continue
    return None

ANALYSIS_PROMPT = """You are a senior biotech investment analyst. Produce a catalyst probability analysis.

{context}

{subfactor_instructions}

USER FOCUS: {question}

STRUCTURE (keep tight, quantitative):

## 1. Company Snapshot
One short paragraph: company, pipeline stage, cash position, key recent moves.

## 2. Catalyst Details
Mechanism, trial design (endpoints, n, comparator), Phase 3 efficacy numbers.

## 3. EVIDENCE TABLE (6-8 rows)
| # | Evidence | Impact | Weight | Source |
|---|----------|--------|--------|--------|
| 1 | Phase 3 efficacy 83.7% | **+** Strong | 25% | Trial readout |
| 2 | Arexvy approved 82.6% | **+** Moderate | 10% | FDA precedent |
Impact: **+** favors approval, **-** against, **?** ambiguous.

## 4. Precedent Comparables (2 approvals + 2 rejections)
| Drug | Indication | Result | Relevance |

## 5. Bull Thesis — WHY IT WILL HIT (4 specific evidence-backed points)

## 6. Bear Thesis — WHY IT WILL MISS (4 specific evidence-backed points)

## 7. Sub-Factor Scores (0-100, one line each)
- Trial Design: X/100 — justification
- Phase 3 Efficacy: X/100 — justification
- Safety Profile: X/100 — justification
- Historical Base Rate: X/100 — justification
- FDA History: X/100 — justification
- Competitive Landscape: X/100 — justification
- Management Execution: X/100 — justification

## 8. VERDICT
One-paragraph synthesis. End with **FINAL PROBABILITY: X%**

Target 400-550 words. Be specific with numbers and named comparables."""

CONSENSUS_PROMPT = """Three biotech analysts analyzed the same catalyst. Produce a fast triangulated consensus.

=== CLAUDE ===
{claude}

=== GEMINI ===
{gemini}

=== GPT ===
{gpt}

Output:

## Agreement: what all 3 agree on (3-5 bullets)

## Disagreement: key differences + who's most credible here (3 bullets)

## Consensus Probability
- Claude: X%, Gemini: Y%, GPT: Z%
- **Consensus: W%** — 2-3 sentence justification

## One-Paragraph Final Verdict

**CONSENSUS FINAL PROBABILITY: W%**

Under 300 words. Fast synthesis — no duplication."""

def _call_claude(prompt, max_tok=2500, feature="ai_pipeline", ticker=None):
    import time as _t
    t0 = _t.time()
    model = "claude-sonnet-4-5"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY",""), timeout=55.0)
        msg = client.messages.create(model=model, max_tokens=max_tok,
            messages=[{"role":"user","content":prompt}])
        try:
            from services.llm_usage import record_usage
            usage = getattr(msg, "usage", None)
            record_usage(provider="anthropic", model=model, feature=feature, ticker=ticker,
                         tokens_input=getattr(usage, "input_tokens", 0) or 0 if usage else 0,
                         tokens_output=getattr(usage, "output_tokens", 0) or 0 if usage else 0,
                         duration_ms=int((_t.time()-t0)*1000), status="success")
        except Exception: pass
        return msg.content[0].text
    except Exception as e:
        try:
            from services.llm_usage import record_usage
            record_usage(provider="anthropic", model=model, feature=feature, ticker=ticker,
                         tokens_input=0, tokens_output=0,
                         duration_ms=int((_t.time()-t0)*1000), status="error",
                         error_message=str(e)[:300])
        except Exception: pass
        return f"**Claude error:** {e}"

def _call_gemini(prompt, feature="ai_pipeline", ticker=None):
    import time as _t
    t0 = _t.time()
    last_err = None
    last_model = None
    for model in ["gemini-2.5-flash", "gemini-2.5-pro"]:
        last_model = model
        for attempt in range(2):
            try:
                from google import genai as google_genai
                client = google_genai.Client(api_key=os.getenv("GOOGLE_API_KEY",""))
                r = client.models.generate_content(model=model, contents=prompt)
                try:
                    from services.llm_usage import record_usage
                    usage = getattr(r, "usage_metadata", None)
                    record_usage(provider="google", model=model, feature=feature, ticker=ticker,
                                 tokens_input=getattr(usage, "prompt_token_count", 0) or 0 if usage else 0,
                                 tokens_output=getattr(usage, "candidates_token_count", 0) or 0 if usage else 0,
                                 duration_ms=int((_t.time()-t0)*1000), status="success")
                except Exception: pass
                return r.text
            except Exception as e:
                last_err = str(e)
                if "503" in last_err or "UNAVAILABLE" in last_err:
                    _time.sleep(2); continue
                else: break
    try:
        from services.llm_usage import record_usage
        record_usage(provider="google", model=last_model, feature=feature, ticker=ticker,
                     tokens_input=0, tokens_output=0,
                     duration_ms=int((_t.time()-t0)*1000), status="error",
                     error_message=(last_err or "")[:300])
    except Exception: pass
    return f"**Gemini error:** {last_err}"

def _call_gpt(prompt, feature="ai_pipeline", ticker=None):
    import time as _t
    t0 = _t.time()
    model = "gpt-4o"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY",""), timeout=55.0)
        resp = client.chat.completions.create(model=model, max_tokens=2500,
            messages=[
                {"role":"system","content":"You are a senior biotech investment analyst. Be quantitative."},
                {"role":"user","content":prompt}])
        try:
            from services.llm_usage import record_usage
            usage = getattr(resp, "usage", None)
            record_usage(provider="openai", model=model, feature=feature, ticker=ticker,
                         tokens_input=getattr(usage, "prompt_tokens", 0) or 0 if usage else 0,
                         tokens_output=getattr(usage, "completion_tokens", 0) or 0 if usage else 0,
                         duration_ms=int((_t.time()-t0)*1000), status="success")
        except Exception: pass
        return resp.choices[0].message.content
    except Exception as e:
        try:
            from services.llm_usage import record_usage
            record_usage(provider="openai", model=model, feature=feature, ticker=ticker,
                         tokens_input=0, tokens_output=0,
                         duration_ms=int((_t.time()-t0)*1000), status="error",
                         error_message=str(e)[:300])
        except Exception: pass
        return f"**GPT error:** {e}"

def run_parallel_only(context, question="", subfactor_weights=None, progress_cb=None):
    """3 models in parallel, NO consensus pass. Return when all 3 done. ~20-30s.
    
    Returns same shape as run_full_pipeline but with consensus=None. UI can call
    compute_consensus() separately to get triangulation later.
    """
    sf_instructions = ""
    if subfactor_weights:
        parts = [f"- {k.replace('_',' ').title()}: weight {v:.0%}" for k,v in subfactor_weights.items()]
        sf_instructions = "User-specified sub-factor weights:\n" + "\n".join(parts)

    prompt = ANALYSIS_PROMPT.format(
        context=context, subfactor_instructions=sf_instructions,
        question=question or "Full catalyst probability analysis")

    t0 = _time.time()
    if progress_cb: progress_cb(5, "🟣🔵🟢 3 models analysing in parallel...")

    results = {}
    OVERALL_TIMEOUT = 70  # seconds
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {
            ex.submit(_call_claude, prompt): "claude",
            ex.submit(_call_gemini, prompt): "gemini",
            ex.submit(_call_gpt, prompt): "gpt",
        }
        done_count = 0
        # Tolerate partial completion: if 1 provider hangs past timeout,
        # use the 2 that finished rather than failing everything.
        try:
            for fut in as_completed(futs, timeout=OVERALL_TIMEOUT):
                name = futs[fut]
                try:
                    results[name] = fut.result(timeout=5)
                    done_count += 1
                    elapsed = _time.time() - t0
                    logger.info(f"  {name} ready at {elapsed:.1f}s")
                    if progress_cb: progress_cb(20 + done_count*25, f"✅ {name} ready at {elapsed:.0f}s")
                except Exception as e:
                    results[name] = f"**{name} failed:** {e}"
                    logger.warning(f"  {name} failed: {e}")
        except FuturesTimeoutError:
            logger.warning(f"parallel: {OVERALL_TIMEOUT}s overall timeout reached — using {done_count}/3 results")
        # After main loop (either completed or timed out): salvage any
        # remaining-but-already-done futures, mark the rest as timed out.
        for fut, name in futs.items():
            if name in results:
                continue
            if fut.done():
                try:
                    results[name] = fut.result(timeout=1)
                    done_count += 1
                    logger.info(f"  {name} salvaged after main loop")
                except Exception as e:
                    results[name] = f"**{name} failed:** {e}"
            else:
                fut.cancel()
                results[name] = f"**{name} failed:** timed out after {OVERALL_TIMEOUT}s and was cancelled"
                logger.warning(f"  {name} cancelled (still running at timeout)")

    claude_out = results.get("claude", "")
    gemini_out = results.get("gemini", "")
    gpt_out    = results.get("gpt", "")
    claude_prob = _extract_probability(claude_out)
    gemini_prob = _extract_probability(gemini_out)
    gpt_prob    = _extract_probability(gpt_out)

    base_m = re.search(r'PROBABILITY:\s*(\d+)%', context)
    base_prob = (int(base_m.group(1))/100.0) if base_m else 0.5

    probs = [p for p in [claude_prob, gemini_prob, gpt_prob] if p is not None]
    mean_prob = sum(probs)/len(probs) if probs else base_prob
    total_time = _time.time()-t0
    logger.info(f"PARALLEL complete: {total_time:.1f}s | mean prob {mean_prob:.0%}")
    if progress_cb: progress_cb(100, f"✅ 3 analyses in {total_time:.0f}s")

    return {
        "draft":            claude_out,
        "gemini_critique":  gemini_out,
        "gpt_critique":     gpt_out,
        "revised":          None,  # consensus not computed yet
        "base_probability": base_prob,
        "ai_probability":   mean_prob,  # Use mean until consensus arrives
        "probabilities_all": {
            "claude_draft": claude_prob,
            "gemini":       gemini_prob,
            "gpt":          gpt_prob,
            "final":        None,
        },
        "elapsed_seconds": total_time,
    }

def compute_consensus(parallel_result, progress_cb=None):
    """Run a fast consensus synthesis on top of existing parallel results. ~10s.
    
    Tries Claude first, falls back to GPT, then Gemini — so if one provider is
    down or out of credits, consensus still renders.
    """
    if parallel_result.get("revised"): return parallel_result
    if progress_cb: progress_cb(10, "🎯 Synthesizing consensus...")
    t0 = _time.time()
    
    prompt = CONSENSUS_PROMPT.format(
        claude=parallel_result["draft"],
        gemini=parallel_result["gemini_critique"],
        gpt=parallel_result["gpt_critique"]
    )
    
    providers = [("Claude", _call_claude, {"max_tok": 1200}),
                 ("GPT", _call_gpt, {}),
                 ("Gemini", _call_gemini, {})]
    
    consensus = None
    last_err = None
    provider_used = None
    for name, fn, kwargs in providers:
        if progress_cb: progress_cb(40, f"🎯 {name} synthesizing...")
        try:
            result = fn(prompt, **kwargs) if kwargs else fn(prompt)
            if result and not result.startswith(f"**{name} error"):
                consensus = result
                provider_used = name
                logger.info(f"Consensus via {name}")
                break
            else:
                last_err = result[:200] if result else f"{name} empty response"
        except Exception as e:
            last_err = f"{name}: {type(e).__name__}: {e}"
            logger.warning(f"Consensus {name} failed: {e}")
    
    if consensus is None:
        # Provide a deterministic fallback: average the 3 probabilities
        probs = parallel_result.get("probabilities_all", {})
        nums = [p for p in [probs.get("claude_draft"), probs.get("gemini"), probs.get("gpt")] if p is not None]
        avg = sum(nums) / len(nums) if nums else None
        consensus = f"""## Consensus unavailable — all 3 providers failed

Last error: {last_err}

## Fallback: arithmetic mean of model probabilities
- Claude: {probs.get('claude_draft','—') if probs.get('claude_draft') is None else f"{probs.get('claude_draft'):.0%}"}
- Gemini: {probs.get('gemini','—') if probs.get('gemini') is None else f"{probs.get('gemini'):.0%}"}
- GPT: {probs.get('gpt','—') if probs.get('gpt') is None else f"{probs.get('gpt'):.0%}"}

**CONSENSUS FINAL PROBABILITY: {int(avg*100)}%** (mean of available models)""" if avg else f"All consensus providers failed. Last error: {last_err}"
        parallel_result["consensus_error"] = last_err
        final_prob = avg
    else:
        final_prob = _extract_probability(consensus)
        parallel_result["consensus_provider"] = provider_used
    
    parallel_result["revised"] = consensus
    parallel_result["probabilities_all"]["final"] = final_prob
    if final_prob is not None:
        parallel_result["ai_probability"] = final_prob
    logger.info(f"Consensus done: {_time.time()-t0:.1f}s via {provider_used or 'FALLBACK'}")
    if progress_cb: progress_cb(100, f"✅ Consensus in {_time.time()-t0:.0f}s")
    return parallel_result

# Keep old name as alias so app.py doesn't break — but run_parallel_only by default
def run_full_pipeline(context, question="", subfactor_weights=None, progress_cb=None):
    """Full pipeline: parallel 3 models + consensus. ~30-45s total."""
    res = run_parallel_only(context, question, subfactor_weights, progress_cb)
    return compute_consensus(res, progress_cb)

def run_quick(context, question="", subfactor_weights=None, progress_cb=None):
    """Single-model quick analysis using Gemini Flash. ~8-12s.
    For sanity checks or when user wants speed over multi-model triangulation."""
    sf_instructions = ""
    if subfactor_weights:
        parts = [f"- {k.replace('_',' ').title()}: weight {v:.0%}" for k,v in subfactor_weights.items()]
        sf_instructions = "User-specified sub-factor weights:\n" + "\n".join(parts)

    prompt = ANALYSIS_PROMPT.format(
        context=context, subfactor_instructions=sf_instructions,
        question=question or "Full catalyst probability analysis")

    t0 = _time.time()
    if progress_cb: progress_cb(20, "🔵 Gemini Flash analysing...")
    result = _call_gemini(prompt)
    prob = _extract_probability(result)
    elapsed = _time.time() - t0
    if progress_cb: progress_cb(100, f"✅ Done in {elapsed:.0f}s")
    logger.info(f"QUICK analysis: {elapsed:.1f}s")

    base_m = re.search(r'PROBABILITY:\s*(\d+)%', context)
    base_prob = (int(base_m.group(1))/100.0) if base_m else 0.5

    return {
        "draft":            result,
        "gemini_critique":  "",
        "gpt_critique":     "",
        "revised":          None,
        "base_probability": base_prob,
        "ai_probability":   prob if prob is not None else base_prob,
        "probabilities_all": {
            "claude_draft": None,
            "gemini":       prob,
            "gpt":          None,
            "final":        None,
        },
        "elapsed_seconds":  elapsed,
        "mode":             "quick",
    }

def run_chat_turn(context, history, question):
    hist_txt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:])
    prompt = f"""Continue the biotech investment analysis dialogue.

DATA CONTEXT:
{context}

CHAT HISTORY:
{hist_txt}

NEW USER QUESTION:
{question}

Answer precisely and quantitatively. Reference specific evidence. If the question is about probability, recalculate and explain the shift."""
    return _call_claude(prompt, max_tok=1200)
