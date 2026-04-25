"""
Universe seeder for Phase B — pulls all biotech/pharma tickers and extracts upcoming catalysts.

Pipeline:
  1. Pull universe candidates (Russell 3000 biotech + Nasdaq Biotech Index + ClinicalTrials.gov sponsors)
  2. For each ticker, extract upcoming non-earnings catalysts in next 6 months
  3. Filter, dedupe, write to catalyst_universe table

Strict cost discipline:
  - Every LLM call gated by LLM_ENABLED env var (default: 'false')
  - When disabled, returns mock data so we can verify the pipes without spending money
  - When enabled, uses gpt-4o-mini exclusively (cheapest model, ~$0.0002/call)
  - Daily spend cap via DAILY_LLM_BUDGET_USD env var (default: $5.00 - intentionally low for safety)

To enable LLM: set LLM_ENABLED=true on the biotech-api Railway service.
"""
import os
import json
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass, asdict

import requests

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────
LLM_ENABLED = os.getenv("LLM_ENABLED", "false").lower() in ("true", "1", "yes")
# Primary: gemini-2.5-flash with Google Search grounding (real-time facts, ~$0.0003/call)
# Fallback: openai gpt-4o (no grounding but better factual recall than mini, ~$0.003/call)
LLM_PROVIDER = os.getenv("UNIVERSE_LLM_PROVIDER", "gemini")  # 'gemini' | 'openai'
LLM_MODEL_GEMINI = os.getenv("UNIVERSE_LLM_MODEL_GEMINI", "gemini-2.5-flash")
LLM_MODEL_OPENAI = os.getenv("UNIVERSE_LLM_MODEL_OPENAI", "gpt-4o")
GEMINI_GROUNDING = os.getenv("UNIVERSE_GROUNDING", "true").lower() in ("true", "1", "yes")
DAILY_LLM_BUDGET_USD = float(os.getenv("DAILY_LLM_BUDGET_USD", "5.00"))
COST_PER_GEMINI_CALL_USD = 0.0003   # gemini-2.5-flash with grounding
COST_PER_OPENAI_CALL_USD = 0.003    # gpt-4o
MAX_TICKERS_PER_RUN = int(os.getenv("MAX_TICKERS_PER_RUN", "50"))  # safety throttle

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# In-memory daily spend tracker (resets on container restart)
_daily_spend = 0.0
_daily_spend_date = None


# ────────────────────────────────────────────────────────────
# Universe candidate sources
# ────────────────────────────────────────────────────────────
def get_seed_universe() -> List[Dict]:
    """
    Returns biotech/pharma ticker candidates from multiple sources.
    
    Phase B v1: starts with a curated seed list of well-known biotechs (~270 tickers).
    Phase B v2: will pull dynamically from SEC EDGAR + Russell 3000 + ClinicalTrials.gov.
    
    Each entry: {ticker, company_name, source}
    """
    # Curated NASDAQ Biotechnology Index members + S&P Pharma — a stable starting point
    # In production, replace with API pulls from SEC EDGAR sector codes
    SEED_TICKERS = [
        # Mega cap pharma (priority)
        {"ticker": "JNJ", "company_name": "Johnson & Johnson", "source": "seed"},
        {"ticker": "LLY", "company_name": "Eli Lilly and Co", "source": "seed"},
        {"ticker": "ABBV", "company_name": "AbbVie Inc", "source": "seed"},
        {"ticker": "MRK", "company_name": "Merck & Co Inc", "source": "seed"},
        {"ticker": "PFE", "company_name": "Pfizer Inc", "source": "seed"},
        {"ticker": "AZN", "company_name": "AstraZeneca PLC", "source": "seed"},
        {"ticker": "NVS", "company_name": "Novartis AG", "source": "seed"},
        {"ticker": "BMY", "company_name": "Bristol-Myers Squibb", "source": "seed"},
        {"ticker": "AMGN", "company_name": "Amgen Inc", "source": "seed"},
        {"ticker": "GILD", "company_name": "Gilead Sciences Inc", "source": "seed"},
        {"ticker": "REGN", "company_name": "Regeneron Pharmaceuticals", "source": "seed"},
        {"ticker": "VRTX", "company_name": "Vertex Pharmaceuticals", "source": "seed"},
        {"ticker": "MRNA", "company_name": "Moderna Inc", "source": "seed"},
        {"ticker": "BNTX", "company_name": "BioNTech SE", "source": "seed"},
        {"ticker": "BIIB", "company_name": "Biogen Inc", "source": "seed"},
        {"ticker": "GSK", "company_name": "GSK plc", "source": "seed"},
        {"ticker": "SNY", "company_name": "Sanofi", "source": "seed"},
        # Mid cap biotechs (priority)
        {"ticker": "ALNY", "company_name": "Alnylam Pharmaceuticals", "source": "seed"},
        {"ticker": "BMRN", "company_name": "BioMarin Pharmaceutical", "source": "seed"},
        {"ticker": "INCY", "company_name": "Incyte Corp", "source": "seed"},
        {"ticker": "EXEL", "company_name": "Exelixis Inc", "source": "seed"},
        {"ticker": "JAZZ", "company_name": "Jazz Pharmaceuticals", "source": "seed"},
        {"ticker": "NBIX", "company_name": "Neurocrine Biosciences", "source": "seed"},
        {"ticker": "SRPT", "company_name": "Sarepta Therapeutics", "source": "seed"},
        {"ticker": "RPRX", "company_name": "Royalty Pharma", "source": "seed"},
        {"ticker": "HALO", "company_name": "Halozyme Therapeutics", "source": "seed"},
        {"ticker": "NTRA", "company_name": "Natera Inc", "source": "seed"},
        # Small cap with active pipelines
        {"ticker": "FOLD", "company_name": "Amicus Therapeutics", "source": "seed"},
        {"ticker": "OCUL", "company_name": "Ocular Therapeutix", "source": "seed"},
        {"ticker": "QNRX", "company_name": "Quoin Pharmaceuticals", "source": "seed"},
        {"ticker": "PRAX", "company_name": "Praxis Precision Medicines", "source": "seed"},
        {"ticker": "IONS", "company_name": "Ionis Pharmaceuticals", "source": "seed"},
        {"ticker": "BCRX", "company_name": "BioCryst Pharmaceuticals", "source": "seed"},
        {"ticker": "ARWR", "company_name": "Arrowhead Pharmaceuticals", "source": "seed"},
        {"ticker": "BLUE", "company_name": "Bluebird Bio Inc", "source": "seed"},
        {"ticker": "ACAD", "company_name": "ACADIA Pharmaceuticals", "source": "seed"},
        {"ticker": "RIGL", "company_name": "Rigel Pharmaceuticals", "source": "seed"},
        {"ticker": "SAGE", "company_name": "Sage Therapeutics", "source": "seed"},
        {"ticker": "TVTX", "company_name": "Travere Therapeutics", "source": "seed"},
        {"ticker": "VKTX", "company_name": "Viking Therapeutics", "source": "seed"},
        {"ticker": "MDGL", "company_name": "Madrigal Pharmaceuticals", "source": "seed"},
        {"ticker": "AXSM", "company_name": "Axsome Therapeutics", "source": "seed"},
        {"ticker": "REPL", "company_name": "Replimune Group", "source": "seed"},
        {"ticker": "INSM", "company_name": "Insmed Inc", "source": "seed"},
        {"ticker": "CRSP", "company_name": "CRISPR Therapeutics", "source": "seed"},
        {"ticker": "NTLA", "company_name": "Intellia Therapeutics", "source": "seed"},
        {"ticker": "BEAM", "company_name": "Beam Therapeutics", "source": "seed"},
        {"ticker": "EDIT", "company_name": "Editas Medicine", "source": "seed"},
        # Tools & diagnostics
        {"ticker": "ILMN", "company_name": "Illumina Inc", "source": "seed"},
        {"ticker": "TMO", "company_name": "Thermo Fisher Scientific", "source": "seed"},
        {"ticker": "DHR", "company_name": "Danaher Corp", "source": "seed"},
        {"ticker": "A", "company_name": "Agilent Technologies", "source": "seed"},
        # ── EXPANDED NASDAQ BIOTECH (228 small/micro caps) ──
        {"ticker": "AGEN", "company_name": "Agenus Inc", "source": "seed"},
        {"ticker": "ANIK", "company_name": "Anika Therapeutics Inc", "source": "seed"},
        {"ticker": "ARQT", "company_name": "Arcutis Biotherapeutics", "source": "seed"},
        {"ticker": "ARDX", "company_name": "Ardelyx Inc", "source": "seed"},
        {"ticker": "ASND", "company_name": "Ascendis Pharma A/S", "source": "seed"},
        {"ticker": "ATAI", "company_name": "ATAI Life Sciences", "source": "seed"},
        {"ticker": "AUPH", "company_name": "Aurinia Pharmaceuticals", "source": "seed"},
        {"ticker": "AVDX", "company_name": "AvidXchange Holdings", "source": "seed"},
        {"ticker": "AVTX", "company_name": "Avalo Therapeutics", "source": "seed"},
        {"ticker": "BBIO", "company_name": "BridgeBio Pharma", "source": "seed"},
        {"ticker": "BPMC", "company_name": "Blueprint Medicines", "source": "seed"},
        {"ticker": "CGEM", "company_name": "Cullinan Therapeutics", "source": "seed"},
        {"ticker": "CGON", "company_name": "CG Oncology", "source": "seed"},
        {"ticker": "COGT", "company_name": "Cogent Biosciences", "source": "seed"},
        {"ticker": "CRDO", "company_name": "Credo Technology", "source": "seed"},
        {"ticker": "CRBP", "company_name": "Corbus Pharmaceuticals", "source": "seed"},
        {"ticker": "CRDF", "company_name": "Cardiff Oncology", "source": "seed"},
        {"ticker": "CRMD", "company_name": "CorMedix Inc", "source": "seed"},
        {"ticker": "CRVS", "company_name": "Corvus Pharmaceuticals", "source": "seed"},
        {"ticker": "CYTK", "company_name": "Cytokinetics Inc", "source": "seed"},
        {"ticker": "DCPH", "company_name": "Deciphera Pharmaceuticals", "source": "seed"},
        {"ticker": "DMAC", "company_name": "DiaMedica Therapeutics", "source": "seed"},
        {"ticker": "DNLI", "company_name": "Denali Therapeutics", "source": "seed"},
        {"ticker": "DRRX", "company_name": "DURECT Corp", "source": "seed"},
        {"ticker": "DVAX", "company_name": "Dynavax Technologies", "source": "seed"},
        {"ticker": "EXAS", "company_name": "Exact Sciences", "source": "seed"},
        {"ticker": "EYPT", "company_name": "EyePoint Pharmaceuticals", "source": "seed"},
        {"ticker": "FATE", "company_name": "Fate Therapeutics", "source": "seed"},
        {"ticker": "FGEN", "company_name": "FibroGen Inc", "source": "seed"},
        {"ticker": "GERN", "company_name": "Geron Corp", "source": "seed"},
        {"ticker": "GMAB", "company_name": "Genmab A/S", "source": "seed"},
        {"ticker": "GTHX", "company_name": "G1 Therapeutics", "source": "seed"},
        {"ticker": "HRMY", "company_name": "Harmony Biosciences", "source": "seed"},
        {"ticker": "HRTX", "company_name": "Heron Therapeutics", "source": "seed"},
        {"ticker": "ICVX", "company_name": "Icosavax Inc", "source": "seed"},
        {"ticker": "IDYA", "company_name": "IDEAYA Biosciences", "source": "seed"},
        {"ticker": "IMCR", "company_name": "Immunocore Holdings", "source": "seed"},
        {"ticker": "IMGN", "company_name": "ImmunoGen Inc", "source": "seed"},
        {"ticker": "IMNM", "company_name": "Immuneering Corp", "source": "seed"},
        {"ticker": "IMUX", "company_name": "Immunic Inc", "source": "seed"},
        {"ticker": "IMVT", "company_name": "Immunovant Inc", "source": "seed"},
        {"ticker": "INMB", "company_name": "INmune Bio", "source": "seed"},
        {"ticker": "INVA", "company_name": "Innoviva Inc", "source": "seed"},
        {"ticker": "IRWD", "company_name": "Ironwood Pharmaceuticals", "source": "seed"},
        {"ticker": "ITCI", "company_name": "Intra-Cellular Therapies", "source": "seed"},
        {"ticker": "ITRM", "company_name": "Iterum Therapeutics", "source": "seed"},
        {"ticker": "IVAC", "company_name": "Intevac Inc", "source": "seed"},
        {"ticker": "JNCE", "company_name": "Jounce Therapeutics", "source": "seed"},
        {"ticker": "KALA", "company_name": "Kala Bio", "source": "seed"},
        {"ticker": "KALV", "company_name": "KalVista Pharmaceuticals", "source": "seed"},
        {"ticker": "KOD", "company_name": "Kodiak Sciences", "source": "seed"},
        {"ticker": "KPTI", "company_name": "Karyopharm Therapeutics", "source": "seed"},
        {"ticker": "KROS", "company_name": "Keros Therapeutics", "source": "seed"},
        {"ticker": "KRTX", "company_name": "Karuna Therapeutics", "source": "seed"},
        {"ticker": "KRYS", "company_name": "Krystal Biotech", "source": "seed"},
        {"ticker": "KURA", "company_name": "Kura Oncology", "source": "seed"},
        {"ticker": "KYMR", "company_name": "Kymera Therapeutics", "source": "seed"},
        {"ticker": "LBPH", "company_name": "Longboard Pharmaceuticals", "source": "seed"},
        {"ticker": "LEGN", "company_name": "Legend Biotech", "source": "seed"},
        {"ticker": "LIAN", "company_name": "LianBio", "source": "seed"},
        {"ticker": "LXRX", "company_name": "Lexicon Pharmaceuticals", "source": "seed"},
        {"ticker": "MGNX", "company_name": "MacroGenics Inc", "source": "seed"},
        {"ticker": "MGTX", "company_name": "MeiraGTx Holdings", "source": "seed"},
        {"ticker": "MIRM", "company_name": "Mirum Pharmaceuticals", "source": "seed"},
        {"ticker": "MNKD", "company_name": "MannKind Corp", "source": "seed"},
        {"ticker": "MNMD", "company_name": "Mind Medicine", "source": "seed"},
        {"ticker": "MORF", "company_name": "Morphic Holding", "source": "seed"},
        {"ticker": "MRSN", "company_name": "Mersana Therapeutics", "source": "seed"},
        {"ticker": "MRTX", "company_name": "Mirati Therapeutics", "source": "seed"},
        {"ticker": "MRUS", "company_name": "Merus N.V.", "source": "seed"},
        {"ticker": "NKTX", "company_name": "Nkarta Inc", "source": "seed"},
        {"ticker": "NUVL", "company_name": "Nuvalent Inc", "source": "seed"},
        {"ticker": "NVAX", "company_name": "Novavax Inc", "source": "seed"},
        {"ticker": "NVTA", "company_name": "Invitae Corp", "source": "seed"},
        {"ticker": "NXTC", "company_name": "NextCure Inc", "source": "seed"},
        {"ticker": "OCGN", "company_name": "Ocugen Inc", "source": "seed"},
        {"ticker": "OGN", "company_name": "Organon & Co", "source": "seed"},
        {"ticker": "OLMA", "company_name": "Olema Pharmaceuticals", "source": "seed"},
        {"ticker": "OMGA", "company_name": "Omega Therapeutics", "source": "seed"},
        {"ticker": "OPTN", "company_name": "OptiNose Inc", "source": "seed"},
        {"ticker": "OYST", "company_name": "Oyster Point Pharma", "source": "seed"},
        {"ticker": "PCRX", "company_name": "Pacira BioSciences", "source": "seed"},
        {"ticker": "PDSB", "company_name": "PDS Biotechnology", "source": "seed"},
        {"ticker": "PHAT", "company_name": "Phathom Pharmaceuticals", "source": "seed"},
        {"ticker": "PHGE", "company_name": "BiomX Inc", "source": "seed"},
        {"ticker": "PIRS", "company_name": "Pieris Pharmaceuticals", "source": "seed"},
        {"ticker": "PRTA", "company_name": "Prothena Corp", "source": "seed"},
        {"ticker": "PSTX", "company_name": "Poseida Therapeutics", "source": "seed"},
        {"ticker": "PTGX", "company_name": "Protagonist Therapeutics", "source": "seed"},
        {"ticker": "RARE", "company_name": "Ultragenyx Pharmaceutical", "source": "seed"},
        {"ticker": "RCKT", "company_name": "Rocket Pharmaceuticals", "source": "seed"},
        {"ticker": "RCM", "company_name": "R1 RCM Inc", "source": "seed"},
        {"ticker": "RCUS", "company_name": "Arcus Biosciences", "source": "seed"},
        {"ticker": "RGNX", "company_name": "REGENXBIO Inc", "source": "seed"},
        {"ticker": "RVMD", "company_name": "Revolution Medicines", "source": "seed"},
        {"ticker": "RXRX", "company_name": "Recursion Pharmaceuticals", "source": "seed"},
        {"ticker": "RYTM", "company_name": "Rhythm Pharmaceuticals", "source": "seed"},
        {"ticker": "SANA", "company_name": "Sana Biotechnology", "source": "seed"},
        {"ticker": "SCPH", "company_name": "scPharmaceuticals", "source": "seed"},
        {"ticker": "SDGR", "company_name": "Schrödinger Inc", "source": "seed"},
        {"ticker": "SEEL", "company_name": "Seelos Therapeutics", "source": "seed"},
        {"ticker": "SGRY", "company_name": "Surgery Partners", "source": "seed"},
        {"ticker": "SLDB", "company_name": "Solid Biosciences", "source": "seed"},
        {"ticker": "SLN", "company_name": "Silence Therapeutics", "source": "seed"},
        {"ticker": "SMMT", "company_name": "Summit Therapeutics", "source": "seed"},
        {"ticker": "SNGX", "company_name": "Soligenix Inc", "source": "seed"},
        {"ticker": "SUPN", "company_name": "Supernus Pharmaceuticals", "source": "seed"},
        {"ticker": "SWAV", "company_name": "ShockWave Medical", "source": "seed"},
        {"ticker": "SWTX", "company_name": "SpringWorks Therapeutics", "source": "seed"},
        {"ticker": "TARS", "company_name": "Tarsus Pharmaceuticals", "source": "seed"},
        {"ticker": "TBPH", "company_name": "Theravance Biopharma", "source": "seed"},
        {"ticker": "TGTX", "company_name": "TG Therapeutics", "source": "seed"},
        {"ticker": "TLRY", "company_name": "Tilray Brands", "source": "seed"},
        {"ticker": "TNYA", "company_name": "Tenaya Therapeutics", "source": "seed"},
        {"ticker": "TPST", "company_name": "Tempest Therapeutics", "source": "seed"},
        {"ticker": "TRDA", "company_name": "Entrada Therapeutics", "source": "seed"},
        {"ticker": "TYRA", "company_name": "TYRA Biosciences", "source": "seed"},
        {"ticker": "UTHR", "company_name": "United Therapeutics", "source": "seed"},
        {"ticker": "VANI", "company_name": "Vivani Medical", "source": "seed"},
        {"ticker": "VERV", "company_name": "Verve Therapeutics", "source": "seed"},
        {"ticker": "VIR", "company_name": "Vir Biotechnology", "source": "seed"},
        {"ticker": "VOR", "company_name": "Vor Biopharma", "source": "seed"},
        {"ticker": "VRDN", "company_name": "Viridian Therapeutics", "source": "seed"},
        {"ticker": "VRNA", "company_name": "Verona Pharma", "source": "seed"},
        {"ticker": "VSTM", "company_name": "Verastem Inc", "source": "seed"},
        {"ticker": "VTYX", "company_name": "Ventyx Biosciences", "source": "seed"},
        {"ticker": "WVE", "company_name": "Wave Life Sciences", "source": "seed"},
        {"ticker": "XBIT", "company_name": "XBiotech Inc", "source": "seed"},
        {"ticker": "XENE", "company_name": "Xenon Pharmaceuticals", "source": "seed"},
        {"ticker": "XERS", "company_name": "Xeris Biopharma", "source": "seed"},
        {"ticker": "XFOR", "company_name": "X4 Pharmaceuticals", "source": "seed"},
        {"ticker": "XOMA", "company_name": "XOMA Royalty Corp", "source": "seed"},
        {"ticker": "ZNTL", "company_name": "Zentalis Pharmaceuticals", "source": "seed"},
        {"ticker": "ZURA", "company_name": "Zura Bio", "source": "seed"},
        {"ticker": "ABCL", "company_name": "AbCellera Biologics", "source": "seed"},
        {"ticker": "ACRS", "company_name": "Aclaris Therapeutics", "source": "seed"},
        {"ticker": "ADAP", "company_name": "Adaptimmune Therapeutics", "source": "seed"},
        {"ticker": "ADMA", "company_name": "ADMA Biologics", "source": "seed"},
        {"ticker": "AGIO", "company_name": "Agios Pharmaceuticals", "source": "seed"},
        {"ticker": "AKBA", "company_name": "Akebia Therapeutics", "source": "seed"},
        {"ticker": "AKRO", "company_name": "Akero Therapeutics", "source": "seed"},
        {"ticker": "ALDX", "company_name": "Aldeyra Therapeutics", "source": "seed"},
        {"ticker": "ALEC", "company_name": "Alector Inc", "source": "seed"},
        {"ticker": "ALKS", "company_name": "Alkermes plc", "source": "seed"},
        {"ticker": "ALLO", "company_name": "Allogene Therapeutics", "source": "seed"},
        {"ticker": "ALVR", "company_name": "AlloVir Inc", "source": "seed"},
        {"ticker": "ALXO", "company_name": "ALX Oncology Holdings", "source": "seed"},
        {"ticker": "AMRN", "company_name": "Amarin Corp plc", "source": "seed"},
        {"ticker": "ANAB", "company_name": "AnaptysBio Inc", "source": "seed"},
        {"ticker": "APLS", "company_name": "Apellis Pharmaceuticals", "source": "seed"},
        {"ticker": "APLT", "company_name": "Applied Therapeutics", "source": "seed"},
        {"ticker": "ARCT", "company_name": "Arcturus Therapeutics", "source": "seed"},
        {"ticker": "ARVN", "company_name": "Arvinas Inc", "source": "seed"},
        {"ticker": "ATRA", "company_name": "Atara Biotherapeutics", "source": "seed"},
        {"ticker": "AVIR", "company_name": "Atea Pharmaceuticals", "source": "seed"},
        {"ticker": "AVXL", "company_name": "Anavex Life Sciences", "source": "seed"},
        {"ticker": "BNGO", "company_name": "Bionano Genomics", "source": "seed"},
        {"ticker": "BTAI", "company_name": "BioXcel Therapeutics", "source": "seed"},
        {"ticker": "BURL", "company_name": "Burlington Stores", "source": "seed"},
        {"ticker": "CABA", "company_name": "Cabaletta Bio", "source": "seed"},
        {"ticker": "CDXS", "company_name": "Codexis Inc", "source": "seed"},
        {"ticker": "CELU", "company_name": "Celularity Inc", "source": "seed"},
        {"ticker": "CHRS", "company_name": "Coherus BioSciences", "source": "seed"},
        {"ticker": "CLDX", "company_name": "Celldex Therapeutics", "source": "seed"},
        {"ticker": "CLVS", "company_name": "Clovis Oncology", "source": "seed"},
        {"ticker": "COLL", "company_name": "Collegium Pharmaceutical", "source": "seed"},
        {"ticker": "CRTX", "company_name": "Cortexyme Inc", "source": "seed"},
        {"ticker": "CTMX", "company_name": "CytomX Therapeutics", "source": "seed"},
        {"ticker": "DARE", "company_name": "Daré Bioscience", "source": "seed"},
        {"ticker": "DAWN", "company_name": "Day One Biopharmaceuticals", "source": "seed"},
        {"ticker": "DCTH", "company_name": "Delcath Systems", "source": "seed"},
        {"ticker": "DERM", "company_name": "Journey Medical Corp", "source": "seed"},
        {"ticker": "DSGN", "company_name": "Design Therapeutics", "source": "seed"},
        {"ticker": "EBS", "company_name": "Emergent BioSolutions", "source": "seed"},
        {"ticker": "ENTA", "company_name": "Enanta Pharmaceuticals", "source": "seed"},
        {"ticker": "ESPR", "company_name": "Esperion Therapeutics", "source": "seed"},
        {"ticker": "ETNB", "company_name": "89bio Inc", "source": "seed"},
        {"ticker": "EVAX", "company_name": "Evaxion Biotech A/S", "source": "seed"},
        {"ticker": "EVGN", "company_name": "Evogene Ltd", "source": "seed"},
        {"ticker": "FBIO", "company_name": "Fortress Biotech", "source": "seed"},
        {"ticker": "FNCH", "company_name": "Finch Therapeutics", "source": "seed"},
        {"ticker": "FRSH", "company_name": "Freshpet Inc", "source": "seed"},
        {"ticker": "FUSN", "company_name": "Fusion Pharmaceuticals", "source": "seed"},
        {"ticker": "GBT", "company_name": "Global Blood Therapeutics", "source": "seed"},
        {"ticker": "GH", "company_name": "Guardant Health", "source": "seed"},
        {"ticker": "GLYC", "company_name": "GlycoMimetics Inc", "source": "seed"},
        {"ticker": "GNFT", "company_name": "GENFIT SA", "source": "seed"},
        {"ticker": "GOSS", "company_name": "Gossamer Bio", "source": "seed"},
        {"ticker": "GTBP", "company_name": "GT Biopharma", "source": "seed"},
        {"ticker": "HZNP", "company_name": "Horizon Therapeutics", "source": "seed"},
        {"ticker": "ICPT", "company_name": "Intercept Pharmaceuticals", "source": "seed"},
        {"ticker": "IDRA", "company_name": "Idera Pharmaceuticals", "source": "seed"},
        {"ticker": "INZY", "company_name": "Inozyme Pharma", "source": "seed"},
        {"ticker": "IPSC", "company_name": "Century Therapeutics", "source": "seed"},
        {"ticker": "ITOS", "company_name": "iTeos Therapeutics", "source": "seed"},
        {"ticker": "JANX", "company_name": "Janux Therapeutics", "source": "seed"},
        {"ticker": "KIDS", "company_name": "OrthoPediatrics Corp", "source": "seed"},
        {"ticker": "KOSA", "company_name": "Kosa Pharmaceuticals", "source": "seed"},
        {"ticker": "LCTX", "company_name": "Lineage Cell Therapeutics", "source": "seed"},
        {"ticker": "LMNL", "company_name": "Liminal BioSciences", "source": "seed"},
        {"ticker": "MERS", "company_name": "Mereo BioPharma", "source": "seed"},
        {"ticker": "MNTK", "company_name": "Montrose Environmental", "source": "seed"},
        {"ticker": "MOR", "company_name": "MorphoSys AG", "source": "seed"},
        {"ticker": "NRBO", "company_name": "NeuroBo Pharmaceuticals", "source": "seed"},
        {"ticker": "OCSL", "company_name": "Oaktree Specialty Lending", "source": "seed"},
        {"ticker": "OMER", "company_name": "Omeros Corp", "source": "seed"},
        {"ticker": "ONCY", "company_name": "Oncolytics Biotech", "source": "seed"},
        {"ticker": "OPK", "company_name": "OPKO Health", "source": "seed"},
        {"ticker": "ORIC", "company_name": "ORIC Pharmaceuticals", "source": "seed"},
        {"ticker": "OSPN", "company_name": "OneSpan Inc", "source": "seed"},
        {"ticker": "PACB", "company_name": "Pacific Biosciences", "source": "seed"},
        {"ticker": "PEPG", "company_name": "PepGen Inc", "source": "seed"},
        {"ticker": "PGEN", "company_name": "Precigen Inc", "source": "seed"},
        {"ticker": "PMVP", "company_name": "PMV Pharmaceuticals", "source": "seed"},
        {"ticker": "PRLD", "company_name": "Prelude Therapeutics", "source": "seed"},
        {"ticker": "PRTC", "company_name": "PureTech Health plc", "source": "seed"},
        {"ticker": "PRVB", "company_name": "Provention Bio", "source": "seed"},
        {"ticker": "PYXS", "company_name": "Pyxis Tankers", "source": "seed"},
        {"ticker": "QURE", "company_name": "uniQure N.V.", "source": "seed"},
        {"ticker": "RGEN", "company_name": "Repligen Corp", "source": "seed"},
        {"ticker": "RKLB", "company_name": "Rocket Lab USA", "source": "seed"},
        {"ticker": "RLAY", "company_name": "Relay Therapeutics", "source": "seed"},
        {"ticker": "RNA", "company_name": "Avidity Biosciences", "source": "seed"},
        {"ticker": "RPTX", "company_name": "Repare Therapeutics", "source": "seed"},
        {"ticker": "SAVA", "company_name": "Cassava Sciences", "source": "seed"},
        {"ticker": "SGMO", "company_name": "Sangamo Therapeutics", "source": "seed"},
        {"ticker": "TWST", "company_name": "Twist Bioscience", "source": "seed"},
        {"ticker": "ZYME", "company_name": "Zymeworks Inc", "source": "seed"},
        # ── FULL BIOTECH UNIVERSE EXPANSION (374 additional small/mid caps) ──
        {"ticker": "ABOS", "company_name": "Acumen Pharmaceuticals", "source": "seed"},
        {"ticker": "ACET", "company_name": "Adicet Bio", "source": "seed"},
        {"ticker": "ACHL", "company_name": "Achilles Therapeutics", "source": "seed"},
        {"ticker": "ACIU", "company_name": "AC Immune SA", "source": "seed"},
        {"ticker": "ACLX", "company_name": "Arcellx Inc", "source": "seed"},
        {"ticker": "ACTU", "company_name": "Actuate Therapeutics", "source": "seed"},
        {"ticker": "ADCT", "company_name": "ADC Therapeutics", "source": "seed"},
        {"ticker": "ADGM", "company_name": "Adagene Inc", "source": "seed"},
        {"ticker": "ADIL", "company_name": "Adial Pharmaceuticals", "source": "seed"},
        {"ticker": "ADXN", "company_name": "Addex Therapeutics", "source": "seed"},
        {"ticker": "AGRX", "company_name": "Agile Therapeutics", "source": "seed"},
        {"ticker": "AKLI", "company_name": "Akili Inc", "source": "seed"},
        {"ticker": "AKTX", "company_name": "Akeso/Akros", "source": "seed"},
        {"ticker": "ALCO", "company_name": "Alcobra/Adlon", "source": "seed"},
        {"ticker": "ALGS", "company_name": "Aligos Therapeutics", "source": "seed"},
        {"ticker": "ALMS", "company_name": "Alumis Inc", "source": "seed"},
        {"ticker": "ALPN", "company_name": "Alpine Immune Sciences", "source": "seed"},
        {"ticker": "ALT", "company_name": "Altimmune Inc", "source": "seed"},
        {"ticker": "ALTO", "company_name": "Alto Neuroscience", "source": "seed"},
        {"ticker": "AMLX", "company_name": "Amylyx Pharmaceuticals", "source": "seed"},
        {"ticker": "AMPE", "company_name": "Ampio Pharmaceuticals", "source": "seed"},
        {"ticker": "AMRX", "company_name": "Amneal Pharmaceuticals", "source": "seed"},
        {"ticker": "AMTX", "company_name": "Aemetis Inc", "source": "seed"},
        {"ticker": "ANEB", "company_name": "Anebulo Pharmaceuticals", "source": "seed"},
        {"ticker": "ANGH", "company_name": "Anghami", "source": "seed"},
        {"ticker": "ANIP", "company_name": "ANI Pharmaceuticals", "source": "seed"},
        {"ticker": "ANIX", "company_name": "Anixa Biosciences", "source": "seed"},
        {"ticker": "ANNX", "company_name": "Annexon Inc", "source": "seed"},
        {"ticker": "AORT", "company_name": "Artivion Inc", "source": "seed"},
        {"ticker": "APGE", "company_name": "Apogee Therapeutics", "source": "seed"},
        {"ticker": "APLM", "company_name": "Apollomics Inc", "source": "seed"},
        {"ticker": "APM", "company_name": "Aptorum Group", "source": "seed"},
        {"ticker": "APRE", "company_name": "Aprea Therapeutics", "source": "seed"},
        {"ticker": "APTO", "company_name": "Aptose Biosciences", "source": "seed"},
        {"ticker": "APYX", "company_name": "Apyx Medical", "source": "seed"},
        {"ticker": "AQB", "company_name": "AquaBounty Technologies", "source": "seed"},
        {"ticker": "AQST", "company_name": "Aquestive Therapeutics", "source": "seed"},
        {"ticker": "ARDS", "company_name": "Aridis Pharmaceuticals", "source": "seed"},
        {"ticker": "ARMP", "company_name": "Armata Pharmaceuticals", "source": "seed"},
        {"ticker": "ARTL", "company_name": "Artelo Biosciences", "source": "seed"},
        {"ticker": "ASMB", "company_name": "Assembly Biosciences", "source": "seed"},
        {"ticker": "ASRT", "company_name": "Assertio Holdings", "source": "seed"},
        {"ticker": "ATEX", "company_name": "Anterix Inc", "source": "seed"},
        {"ticker": "ATHA", "company_name": "Athira Pharma", "source": "seed"},
        {"ticker": "ATHX", "company_name": "Athersys Inc", "source": "seed"},
        {"ticker": "ATNM", "company_name": "Actinium Pharmaceuticals", "source": "seed"},
        {"ticker": "ATNF", "company_name": "180 Life Sciences", "source": "seed"},
        {"ticker": "ATOS", "company_name": "Atossa Therapeutics", "source": "seed"},
        {"ticker": "ATXI", "company_name": "Avenue Therapeutics", "source": "seed"},
        {"ticker": "AUST", "company_name": "Austin Gold", "source": "seed"},
        {"ticker": "AVAH", "company_name": "Aveanna Healthcare", "source": "seed"},
        {"ticker": "AVDL", "company_name": "Avadel Pharmaceuticals", "source": "seed"},
        {"ticker": "AVRO", "company_name": "AVROBIO Inc", "source": "seed"},
        {"ticker": "AVTE", "company_name": "Aerovate Therapeutics", "source": "seed"},
        {"ticker": "AXGN", "company_name": "AxoGen Inc", "source": "seed"},
        {"ticker": "AXLA", "company_name": "Axcella Health", "source": "seed"},
        {"ticker": "AZRX", "company_name": "AzurRx BioPharma", "source": "seed"},
        {"ticker": "BCAB", "company_name": "BioAtla Inc", "source": "seed"},
        {"ticker": "BCDA", "company_name": "BioCardia Inc", "source": "seed"},
        {"ticker": "BCLI", "company_name": "Brainstorm Cell Therapeutics", "source": "seed"},
        {"ticker": "BCYC", "company_name": "Bicycle Therapeutics", "source": "seed"},
        {"ticker": "BDSI", "company_name": "BioDelivery Sciences", "source": "seed"},
        {"ticker": "BDTX", "company_name": "Black Diamond Therapeutics", "source": "seed"},
        {"ticker": "BDX", "company_name": "Becton Dickinson", "source": "seed"},
        {"ticker": "BFRI", "company_name": "Biofrontera Inc", "source": "seed"},
        {"ticker": "BGNE", "company_name": "BeiGene Ltd", "source": "seed"},
        {"ticker": "BHC", "company_name": "Bausch Health", "source": "seed"},
        {"ticker": "BHVN", "company_name": "Biohaven Ltd", "source": "seed"},
        {"ticker": "BIVI", "company_name": "BioVie Inc", "source": "seed"},
        {"ticker": "BIOR", "company_name": "Biora Therapeutics", "source": "seed"},
        {"ticker": "BLCM", "company_name": "Bellicum Pharmaceuticals", "source": "seed"},
        {"ticker": "BLRX", "company_name": "BioLineRx Ltd", "source": "seed"},
        {"ticker": "BMEA", "company_name": "Biomea Fusion", "source": "seed"},
        {"ticker": "BNAI", "company_name": "Brand Engagement Network", "source": "seed"},
        {"ticker": "BNTC", "company_name": "Benitec Biopharma", "source": "seed"},
        {"ticker": "BOLT", "company_name": "Bolt Biotherapeutics", "source": "seed"},
        {"ticker": "BVS", "company_name": "Bioventus Inc", "source": "seed"},
        {"ticker": "BWAY", "company_name": "BrainsWay Ltd", "source": "seed"},
        {"ticker": "CALT", "company_name": "Calliditas Therapeutics", "source": "seed"},
        {"ticker": "CAPR", "company_name": "Capricor Therapeutics", "source": "seed"},
        {"ticker": "CARA", "company_name": "Cara Therapeutics", "source": "seed"},
        {"ticker": "CARM", "company_name": "Carisma Therapeutics", "source": "seed"},
        {"ticker": "CASI", "company_name": "CASI Pharmaceuticals", "source": "seed"},
        {"ticker": "CBAY", "company_name": "CymaBay Therapeutics", "source": "seed"},
        {"ticker": "CBIO", "company_name": "Catalyst Biosciences", "source": "seed"},
        {"ticker": "CDMO", "company_name": "Avid Bioservices", "source": "seed"},
        {"ticker": "CDNA", "company_name": "CareDx Inc", "source": "seed"},
        {"ticker": "CDTX", "company_name": "Cidara Therapeutics", "source": "seed"},
        {"ticker": "CDXC", "company_name": "ChromaDex Corp", "source": "seed"},
        {"ticker": "CEAD", "company_name": "CEA Industries", "source": "seed"},
        {"ticker": "CELC", "company_name": "Celcuity Inc", "source": "seed"},
        {"ticker": "CERS", "company_name": "Cerus Corp", "source": "seed"},
        {"ticker": "CERT", "company_name": "Certara Inc", "source": "seed"},
        {"ticker": "CFRX", "company_name": "ContraFect Corp", "source": "seed"},
        {"ticker": "CKPT", "company_name": "Checkpoint Therapeutics", "source": "seed"},
        {"ticker": "CLLS", "company_name": "Cellectis SA", "source": "seed"},
        {"ticker": "CLNN", "company_name": "Clene Inc", "source": "seed"},
        {"ticker": "CLPS", "company_name": "CLPS Inc", "source": "seed"},
        {"ticker": "CLRB", "company_name": "Cellectar Biosciences", "source": "seed"},
        {"ticker": "CLSD", "company_name": "Clearside Biomedical", "source": "seed"},
        {"ticker": "CMMB", "company_name": "Chemomab Therapeutics", "source": "seed"},
        {"ticker": "CMPS", "company_name": "Compass Pathways", "source": "seed"},
        {"ticker": "CMRX", "company_name": "Chimerix Inc", "source": "seed"},
        {"ticker": "CNCE", "company_name": "Concert Pharmaceuticals", "source": "seed"},
        {"ticker": "CNTA", "company_name": "Centessa Pharmaceuticals", "source": "seed"},
        {"ticker": "CNTX", "company_name": "Context Therapeutics", "source": "seed"},
        {"ticker": "COCP", "company_name": "Cocrystal Pharma", "source": "seed"},
        {"ticker": "CODX", "company_name": "Co-Diagnostics", "source": "seed"},
        {"ticker": "COMM", "company_name": "Commscope Holding", "source": "seed"},
        {"ticker": "COR", "company_name": "Cencora Inc", "source": "seed"},
        {"ticker": "CPIX", "company_name": "Cumberland Pharmaceuticals", "source": "seed"},
        {"ticker": "CPRX", "company_name": "Catalyst Pharmaceuticals", "source": "seed"},
        {"ticker": "CRDL", "company_name": "Cardiol Therapeutics", "source": "seed"},
        {"ticker": "CRNX", "company_name": "Crinetics Pharmaceuticals", "source": "seed"},
        {"ticker": "CRTO", "company_name": "Criteo SA", "source": "seed"},
        {"ticker": "CRVL", "company_name": "CorVel Corp", "source": "seed"},
        {"ticker": "CSTL", "company_name": "Castle Biosciences", "source": "seed"},
        {"ticker": "CTKB", "company_name": "Cytek Biosciences", "source": "seed"},
        {"ticker": "CTLT", "company_name": "Catalent Inc", "source": "seed"},
        {"ticker": "CUE", "company_name": "Cue Biopharma", "source": "seed"},
        {"ticker": "CULL", "company_name": "Cullinan Oncology", "source": "seed"},
        {"ticker": "CUTR", "company_name": "Cutera Inc", "source": "seed"},
        {"ticker": "CVAC", "company_name": "CureVac NV", "source": "seed"},
        {"ticker": "CVM", "company_name": "CEL-SCI Corp", "source": "seed"},
        {"ticker": "CYAN", "company_name": "Cyanotech Corp", "source": "seed"},
        {"ticker": "CYCC", "company_name": "Cyclacel Pharmaceuticals", "source": "seed"},
        {"ticker": "CYTH", "company_name": "Cyngn Inc", "source": "seed"},
        {"ticker": "DAIO", "company_name": "Data I/O Corp", "source": "seed"},
        {"ticker": "DENN", "company_name": "Denny's Corp", "source": "seed"},
        {"ticker": "DGX", "company_name": "Quest Diagnostics", "source": "seed"},
        {"ticker": "DHIL", "company_name": "Diamond Hill", "source": "seed"},
        {"ticker": "DIFF", "company_name": "Hadrian Inc", "source": "seed"},
        {"ticker": "DKNG", "company_name": "DraftKings", "source": "seed"},
        {"ticker": "DMTK", "company_name": "DermTech Inc", "source": "seed"},
        {"ticker": "DRUG", "company_name": "Bright Minds Biosciences", "source": "seed"},
        {"ticker": "DTIL", "company_name": "Precision BioSciences", "source": "seed"},
        {"ticker": "DXCM", "company_name": "DexCom Inc", "source": "seed"},
        {"ticker": "DYAI", "company_name": "Dyadic International", "source": "seed"},
        {"ticker": "DYN", "company_name": "Dyne Therapeutics", "source": "seed"},
        {"ticker": "EARS", "company_name": "Auris Medical", "source": "seed"},
        {"ticker": "EBET", "company_name": "EBET Inc", "source": "seed"},
        {"ticker": "EDAP", "company_name": "EDAP TMS", "source": "seed"},
        {"ticker": "EFTR", "company_name": "eFFECTOR Therapeutics", "source": "seed"},
        {"ticker": "ELOX", "company_name": "Eloxx Pharmaceuticals", "source": "seed"},
        {"ticker": "ELTX", "company_name": "Elicio Therapeutics", "source": "seed"},
        {"ticker": "ELVA", "company_name": "Electra Battery Materials", "source": "seed"},
        {"ticker": "ELVN", "company_name": "Enliven Therapeutics", "source": "seed"},
        {"ticker": "ENB", "company_name": "Enbridge", "source": "seed"},
        {"ticker": "ENLV", "company_name": "Enlivex Therapeutics", "source": "seed"},
        {"ticker": "ENSC", "company_name": "Ensysce Biosciences", "source": "seed"},
        {"ticker": "ENTX", "company_name": "Entera Bio", "source": "seed"},
        {"ticker": "EOLS", "company_name": "Evolus Inc", "source": "seed"},
        {"ticker": "ERAS", "company_name": "Erasca Inc", "source": "seed"},
        {"ticker": "ETON", "company_name": "Eton Pharmaceuticals", "source": "seed"},
        {"ticker": "EVH", "company_name": "Evolent Health", "source": "seed"},
        {"ticker": "EVLO", "company_name": "Evelo Biosciences", "source": "seed"},
        {"ticker": "EVOK", "company_name": "Evoke Pharma", "source": "seed"},
        {"ticker": "EYEN", "company_name": "Eyenovia Inc", "source": "seed"},
        {"ticker": "FBRX", "company_name": "Forte Biosciences", "source": "seed"},
        {"ticker": "FENC", "company_name": "Fennec Pharmaceuticals", "source": "seed"},
        {"ticker": "FOSL", "company_name": "Fossil Group", "source": "seed"},
        {"ticker": "FPI", "company_name": "Farmland Partners", "source": "seed"},
        {"ticker": "FRGE", "company_name": "Forge Global Holdings", "source": "seed"},
        {"ticker": "FREQ", "company_name": "Frequency Therapeutics", "source": "seed"},
        {"ticker": "GALT", "company_name": "Galectin Therapeutics", "source": "seed"},
        {"ticker": "GLPG", "company_name": "Galapagos NV", "source": "seed"},
        {"ticker": "GRTX", "company_name": "Galera Therapeutics", "source": "seed"},
        {"ticker": "HBIO", "company_name": "Harvard Bioscience", "source": "seed"},
        {"ticker": "HCM", "company_name": "HUTCHMED China Limited", "source": "seed"},
        {"ticker": "HCSG", "company_name": "Healthcare Services", "source": "seed"},
        {"ticker": "HEPA", "company_name": "Hepion Pharmaceuticals", "source": "seed"},
        {"ticker": "HGEN", "company_name": "Humanigen Inc", "source": "seed"},
        {"ticker": "HOOK", "company_name": "HOOKIPA Pharma", "source": "seed"},
        {"ticker": "HSDT", "company_name": "Helius Medical", "source": "seed"},
        {"ticker": "HUMA", "company_name": "Humacyte Inc", "source": "seed"},
        {"ticker": "ICCC", "company_name": "ImmuCell Corp", "source": "seed"},
        {"ticker": "ICCM", "company_name": "Icecure Medical", "source": "seed"},
        {"ticker": "IDXX", "company_name": "IDEXX Laboratories", "source": "seed"},
        {"ticker": "IGMS", "company_name": "IGM Biosciences", "source": "seed"},
        {"ticker": "IHRT", "company_name": "iHeartMedia", "source": "seed"},
        {"ticker": "IKNA", "company_name": "Ikena Oncology", "source": "seed"},
        {"ticker": "IMAB", "company_name": "I-Mab", "source": "seed"},
        {"ticker": "IMMP", "company_name": "Immutep Limited", "source": "seed"},
        {"ticker": "IMRX", "company_name": "Immuneon Therapeutics", "source": "seed"},
        {"ticker": "INAB", "company_name": "IN8bio Inc", "source": "seed"},
        {"ticker": "INDV", "company_name": "Indivior PLC", "source": "seed"},
        {"ticker": "INMD", "company_name": "InMode Ltd", "source": "seed"},
        {"ticker": "INO", "company_name": "Inovio Pharmaceuticals", "source": "seed"},
        {"ticker": "INTA", "company_name": "Intapp Inc", "source": "seed"},
        {"ticker": "IOBT", "company_name": "IO Biotech", "source": "seed"},
        {"ticker": "IOVA", "company_name": "Iovance Biotherapeutics", "source": "seed"},
        {"ticker": "IRIX", "company_name": "IRIDEX Corp", "source": "seed"},
        {"ticker": "JAGX", "company_name": "Jaguar Health", "source": "seed"},
        {"ticker": "KDNY", "company_name": "Chinook Therapeutics", "source": "seed"},
        {"ticker": "KMDA", "company_name": "Kamada Ltd", "source": "seed"},
        {"ticker": "KNSA", "company_name": "Kiniksa Pharmaceuticals", "source": "seed"},
        {"ticker": "KPRX", "company_name": "Kiromic BioPharma", "source": "seed"},
        {"ticker": "LAB", "company_name": "Standard BioTools", "source": "seed"},
        {"ticker": "LFCR", "company_name": "Lifecore Biomedical", "source": "seed"},
        {"ticker": "LIFE", "company_name": "aTyr Pharma", "source": "seed"},
        {"ticker": "LJPC", "company_name": "La Jolla Pharmaceutical", "source": "seed"},
        {"ticker": "LNTH", "company_name": "Lantheus Holdings", "source": "seed"},
        {"ticker": "LPCN", "company_name": "Lipocine Inc", "source": "seed"},
        {"ticker": "LPTX", "company_name": "Leap Therapeutics", "source": "seed"},
        {"ticker": "LQDA", "company_name": "Liquidia Corp", "source": "seed"},
        {"ticker": "LRMR", "company_name": "Larimar Therapeutics", "source": "seed"},
        {"ticker": "LXEH", "company_name": "Lixiang Education", "source": "seed"},
        {"ticker": "MASS", "company_name": "908 Devices", "source": "seed"},
        {"ticker": "MBIO", "company_name": "Mustang Bio", "source": "seed"},
        {"ticker": "MBOT", "company_name": "Microbot Medical", "source": "seed"},
        {"ticker": "MBRX", "company_name": "Moleculin Biotech", "source": "seed"},
        {"ticker": "MCRB", "company_name": "Seres Therapeutics", "source": "seed"},
        {"ticker": "MDXG", "company_name": "MiMedx Group", "source": "seed"},
        {"ticker": "MERC", "company_name": "Mercer International", "source": "seed"},
        {"ticker": "MGTA", "company_name": "Magenta Therapeutics", "source": "seed"},
        {"ticker": "MLAB", "company_name": "Mesa Labs", "source": "seed"},
        {"ticker": "MNOV", "company_name": "MediciNova", "source": "seed"},
        {"ticker": "MREO", "company_name": "Mereo BioPharma", "source": "seed"},
        {"ticker": "MTEM", "company_name": "Molecular Templates", "source": "seed"},
        {"ticker": "MYGN", "company_name": "Myriad Genetics", "source": "seed"},
        {"ticker": "NEPH", "company_name": "Nephros Inc", "source": "seed"},
        {"ticker": "NEUE", "company_name": "NeuExtra", "source": "seed"},
        {"ticker": "NEWT", "company_name": "NewtekOne", "source": "seed"},
        {"ticker": "NGM", "company_name": "NGM Biopharmaceuticals", "source": "seed"},
        {"ticker": "NKTR", "company_name": "Nektar Therapeutics", "source": "seed"},
        {"ticker": "NMRK", "company_name": "Newmark Group", "source": "seed"},
        {"ticker": "NMTC", "company_name": "NeuroOne Medical", "source": "seed"},
        {"ticker": "NNVC", "company_name": "NanoViricides", "source": "seed"},
        {"ticker": "NOTV", "company_name": "Inotiv Inc", "source": "seed"},
        {"ticker": "NRIX", "company_name": "Nurix Therapeutics", "source": "seed"},
        {"ticker": "NSPR", "company_name": "InspireMD Inc", "source": "seed"},
        {"ticker": "NUWE", "company_name": "Nuwellis Inc", "source": "seed"},
        {"ticker": "NVCR", "company_name": "NovoCure Limited", "source": "seed"},
        {"ticker": "NVCT", "company_name": "Nuvectis Pharma", "source": "seed"},
        {"ticker": "NXPL", "company_name": "Next-ChemX", "source": "seed"},
        {"ticker": "OBLG", "company_name": "Oblong Inc", "source": "seed"},
        {"ticker": "OCEA", "company_name": "Ocean Biomedical", "source": "seed"},
        {"ticker": "ODT", "company_name": "Odonate Therapeutics", "source": "seed"},
        {"ticker": "OGI", "company_name": "Organigram Holdings", "source": "seed"},
        {"ticker": "OMCL", "company_name": "Omnicell Inc", "source": "seed"},
        {"ticker": "ONCT", "company_name": "Oncternal Therapeutics", "source": "seed"},
        {"ticker": "OPGN", "company_name": "OpGen Inc", "source": "seed"},
        {"ticker": "OPRX", "company_name": "OptimizeRx Corp", "source": "seed"},
        {"ticker": "ORGO", "company_name": "Organogenesis", "source": "seed"},
        {"ticker": "ORMP", "company_name": "Oramed Pharmaceuticals", "source": "seed"},
        {"ticker": "OTLK", "company_name": "Outlook Therapeutics", "source": "seed"},
        {"ticker": "OVID", "company_name": "Ovid Therapeutics", "source": "seed"},
        {"ticker": "OWLT", "company_name": "Owlet Inc", "source": "seed"},
        {"ticker": "PALI", "company_name": "Palatin Technologies", "source": "seed"},
        {"ticker": "PAVS", "company_name": "Paltalk Inc", "source": "seed"},
        {"ticker": "PBYI", "company_name": "Puma Biotechnology", "source": "seed"},
        {"ticker": "PETS", "company_name": "PetMed Express", "source": "seed"},
        {"ticker": "PGNY", "company_name": "Progyny Inc", "source": "seed"},
        {"ticker": "PKI", "company_name": "PerkinElmer", "source": "seed"},
        {"ticker": "PLRX", "company_name": "Pliant Therapeutics", "source": "seed"},
        {"ticker": "PNT", "company_name": "POINT Biopharma", "source": "seed"},
        {"ticker": "PODD", "company_name": "Insulet Corp", "source": "seed"},
        {"ticker": "POLY", "company_name": "Plot Twist Capital", "source": "seed"},
        {"ticker": "PRAA", "company_name": "PRA Group", "source": "seed"},
        {"ticker": "PRDS", "company_name": "Pardes Biosciences", "source": "seed"},
        {"ticker": "PRGO", "company_name": "Perrigo Co", "source": "seed"},
        {"ticker": "PRME", "company_name": "Prime Medicine", "source": "seed"},
        {"ticker": "PRPH", "company_name": "ProPhase Labs", "source": "seed"},
        {"ticker": "PRQR", "company_name": "ProQR Therapeutics", "source": "seed"},
        {"ticker": "PRTH", "company_name": "Priority Technology", "source": "seed"},
        {"ticker": "PRTK", "company_name": "Paratek Pharmaceuticals", "source": "seed"},
        {"ticker": "PRTS", "company_name": "CarParts.com", "source": "seed"},
        {"ticker": "PSNL", "company_name": "Personalis Inc", "source": "seed"},
        {"ticker": "PSTV", "company_name": "Plus Therapeutics", "source": "seed"},
        {"ticker": "PTCT", "company_name": "PTC Therapeutics", "source": "seed"},
        {"ticker": "PTPI", "company_name": "Petros Pharmaceuticals", "source": "seed"},
        {"ticker": "QGEN", "company_name": "Qiagen NV", "source": "seed"},
        {"ticker": "QSI", "company_name": "Quanterix Corp", "source": "seed"},
        {"ticker": "RAPP", "company_name": "Rapport Therapeutics", "source": "seed"},
        {"ticker": "RCEL", "company_name": "AVITA Medical", "source": "seed"},
        {"ticker": "RDHL", "company_name": "RedHill Biopharma", "source": "seed"},
        {"ticker": "RDIB", "company_name": "Riley Exploration Permian", "source": "seed"},
        {"ticker": "RETA", "company_name": "Reata Pharmaceuticals", "source": "seed"},
        {"ticker": "REVB", "company_name": "Revelation Biosciences", "source": "seed"},
        {"ticker": "RGC", "company_name": "Regencell Bioscience", "source": "seed"},
        {"ticker": "RLMD", "company_name": "Relmada Therapeutics", "source": "seed"},
        {"ticker": "RLYB", "company_name": "Rallybio Corp", "source": "seed"},
        {"ticker": "RNAZ", "company_name": "TransCode Therapeutics", "source": "seed"},
        {"ticker": "RNXT", "company_name": "RenovoRx Inc", "source": "seed"},
        {"ticker": "ROIV", "company_name": "Roivant Sciences", "source": "seed"},
        {"ticker": "ROOT", "company_name": "Root Inc", "source": "seed"},
        {"ticker": "RVNC", "company_name": "Revance Therapeutics", "source": "seed"},
        {"ticker": "RVPH", "company_name": "Reviva Pharmaceuticals", "source": "seed"},
        {"ticker": "SBPH", "company_name": "Spring Bank Pharmaceuticals", "source": "seed"},
        {"ticker": "SEER", "company_name": "Seer Inc", "source": "seed"},
        {"ticker": "SELB", "company_name": "Selecta Biosciences", "source": "seed"},
        {"ticker": "SENS", "company_name": "Senseonics Holdings", "source": "seed"},
        {"ticker": "SGRP", "company_name": "SPAR Group", "source": "seed"},
        {"ticker": "SHPH", "company_name": "Shuhai Pharmaceuticals", "source": "seed"},
        {"ticker": "SIGA", "company_name": "SIGA Technologies", "source": "seed"},
        {"ticker": "SLNG", "company_name": "Stabilis Solutions", "source": "seed"},
        {"ticker": "SLNO", "company_name": "Soleno Therapeutics", "source": "seed"},
        {"ticker": "SNCY", "company_name": "Sun Country Airlines", "source": "seed"},
        {"ticker": "SNDL", "company_name": "SNDL Inc", "source": "seed"},
        {"ticker": "SNDX", "company_name": "Syndax Pharmaceuticals", "source": "seed"},
        {"ticker": "SNPS", "company_name": "Synopsys Inc", "source": "seed"},
        {"ticker": "SOLO", "company_name": "Electrameccanica Vehicles", "source": "seed"},
        {"ticker": "SONM", "company_name": "SonimTech", "source": "seed"},
        {"ticker": "SOPH", "company_name": "SOPHiA Genetics", "source": "seed"},
        {"ticker": "SPRO", "company_name": "Spero Therapeutics", "source": "seed"},
        {"ticker": "SQNS", "company_name": "Sequans Communications", "source": "seed"},
        {"ticker": "SRDX", "company_name": "Surmodics Inc", "source": "seed"},
        {"ticker": "SRNE", "company_name": "Sorrento Therapeutics", "source": "seed"},
        {"ticker": "SRRK", "company_name": "Scholar Rock Holding", "source": "seed"},
        {"ticker": "SSY", "company_name": "SunLink Health", "source": "seed"},
        {"ticker": "STIM", "company_name": "Neuronetics Inc", "source": "seed"},
        {"ticker": "STOK", "company_name": "Stoke Therapeutics", "source": "seed"},
        {"ticker": "SVRA", "company_name": "Savara Inc", "source": "seed"},
        {"ticker": "SWBI", "company_name": "Smith & Wesson", "source": "seed"},
        {"ticker": "SY", "company_name": "So-Young International", "source": "seed"},
        {"ticker": "SYBX", "company_name": "Synlogic Inc", "source": "seed"},
        {"ticker": "SYK", "company_name": "Stryker Corp", "source": "seed"},
        {"ticker": "SYRS", "company_name": "Syros Pharmaceuticals", "source": "seed"},
        {"ticker": "TARO", "company_name": "Taro Pharmaceutical", "source": "seed"},
        {"ticker": "TBRG", "company_name": "Tobira Therapeutics", "source": "seed"},
        {"ticker": "TCMD", "company_name": "Tactile Systems Technology", "source": "seed"},
        {"ticker": "TCRT", "company_name": "Alaunos Therapeutics", "source": "seed"},
        {"ticker": "TCRX", "company_name": "TCR2 Therapeutics", "source": "seed"},
        {"ticker": "TECH", "company_name": "Bio-Techne Corp", "source": "seed"},
        {"ticker": "TELA", "company_name": "TELA Bio", "source": "seed"},
        {"ticker": "TENX", "company_name": "Tenax Therapeutics", "source": "seed"},
        {"ticker": "TERN", "company_name": "Terns Pharmaceuticals", "source": "seed"},
        {"ticker": "TFFP", "company_name": "TFF Pharmaceuticals", "source": "seed"},
        {"ticker": "THRX", "company_name": "Theseus Pharmaceuticals", "source": "seed"},
        {"ticker": "TIGO", "company_name": "Millicom International", "source": "seed"},
        {"ticker": "TIL", "company_name": "Instil Bio", "source": "seed"},
        {"ticker": "TLIS", "company_name": "Talis Biomedical", "source": "seed"},
        {"ticker": "TNGX", "company_name": "Tango Therapeutics", "source": "seed"},
        {"ticker": "TRIB", "company_name": "Trinity Biotech", "source": "seed"},
        {"ticker": "TRIL", "company_name": "Trillium Therapeutics", "source": "seed"},
        {"ticker": "TRVI", "company_name": "Trevi Therapeutics", "source": "seed"},
        {"ticker": "TRVN", "company_name": "Trevena Inc", "source": "seed"},
        {"ticker": "TSHA", "company_name": "Taysha Gene Therapies", "source": "seed"},
        {"ticker": "TSVT", "company_name": "2seventy bio", "source": "seed"},
        {"ticker": "UMC", "company_name": "United Microelectronics", "source": "seed"},
        {"ticker": "UPB", "company_name": "Upstart Holdings", "source": "seed"},
        {"ticker": "URGN", "company_name": "UroGen Pharma", "source": "seed"},
        {"ticker": "USPH", "company_name": "U.S. Physical Therapy", "source": "seed"},
        {"ticker": "VCEL", "company_name": "Vericel Corp", "source": "seed"},
        {"ticker": "VCNX", "company_name": "Vaccinex Inc", "source": "seed"},
        {"ticker": "VECT", "company_name": "Vector Group", "source": "seed"},
        {"ticker": "VECTOR", "company_name": "Vector", "source": "seed"},
        {"ticker": "VEEV", "company_name": "Veeva Systems", "source": "seed"},
        {"ticker": "VERA", "company_name": "Vera Therapeutics", "source": "seed"},
        {"ticker": "VERI", "company_name": "Veritone Inc", "source": "seed"},
        {"ticker": "VHC", "company_name": "VirnetX Holding", "source": "seed"},
        {"ticker": "VIGL", "company_name": "Vigil Neuroscience", "source": "seed"},
        {"ticker": "VINC", "company_name": "Vincerx Pharma", "source": "seed"},
        {"ticker": "VIRI", "company_name": "Virios Therapeutics", "source": "seed"},
        {"ticker": "VNDA", "company_name": "Vanda Pharmaceuticals", "source": "seed"},
        {"ticker": "VSAT", "company_name": "Viasat", "source": "seed"},
        {"ticker": "VTGN", "company_name": "VistaGen Therapeutics", "source": "seed"},
        {"ticker": "VTRS", "company_name": "Viatris Inc", "source": "seed"},
        {"ticker": "VVOS", "company_name": "Vivos Therapeutics", "source": "seed"},
        {"ticker": "VVPR", "company_name": "VivoPower", "source": "seed"},
        {"ticker": "VYNE", "company_name": "VYNE Therapeutics", "source": "seed"},
        {"ticker": "WLDS", "company_name": "Wearable Devices", "source": "seed"},
        {"ticker": "WULF", "company_name": "TeraWulf Inc", "source": "seed"},
        {"ticker": "X4PH", "company_name": "X4 Pharmaceuticals", "source": "seed"},
        {"ticker": "XAIR", "company_name": "Beyond Air", "source": "seed"},
        {"ticker": "XBIO", "company_name": "Xenetic Biosciences", "source": "seed"},
        {"ticker": "XCUR", "company_name": "Exicure Inc", "source": "seed"},
        {"ticker": "XGN", "company_name": "Exagen Inc", "source": "seed"},
        {"ticker": "XLO", "company_name": "Xilio Therapeutics", "source": "seed"},
        {"ticker": "XRX", "company_name": "Xerox Holdings", "source": "seed"},
        {"ticker": "YMAB", "company_name": "Y-mAbs Therapeutics", "source": "seed"},
        {"ticker": "ZBIO", "company_name": "Zenas BioPharma", "source": "seed"},
        {"ticker": "ZIMV", "company_name": "ZimVie Inc", "source": "seed"},
        {"ticker": "ZYXI", "company_name": "Zynex Medical", "source": "seed"},
    
    ]
    return SEED_TICKERS


# ────────────────────────────────────────────────────────────
# LLM gate (only path that costs money)
# ────────────────────────────────────────────────────────────
def _check_budget() -> bool:
    """Returns True if we have budget remaining today."""
    global _daily_spend, _daily_spend_date
    today = date.today()
    if _daily_spend_date != today:
        _daily_spend = 0.0
        _daily_spend_date = today
    return _daily_spend < DAILY_LLM_BUDGET_USD


def _record_spend(usd: float):
    """Track LLM spend."""
    global _daily_spend, _daily_spend_date
    today = date.today()
    if _daily_spend_date != today:
        _daily_spend = 0.0
        _daily_spend_date = today
    _daily_spend += usd


def get_daily_spend() -> Dict:
    """Inspectable spend stats."""
    return {
        "date": str(_daily_spend_date) if _daily_spend_date else None,
        "spent_usd": round(_daily_spend, 4),
        "budget_usd": DAILY_LLM_BUDGET_USD,
        "remaining_usd": round(DAILY_LLM_BUDGET_USD - _daily_spend, 4),
        "llm_enabled": LLM_ENABLED,
        "provider": LLM_PROVIDER,
        "model_primary": LLM_MODEL_GEMINI if LLM_PROVIDER == "gemini" else LLM_MODEL_OPENAI,
        "model_fallback": LLM_MODEL_OPENAI,
        "grounding": GEMINI_GROUNDING,
    }


def extract_catalysts_for_ticker(ticker: str, company_name: str) -> Tuple[List[Dict], Dict]:
    """
    Extract upcoming catalysts for a single ticker.
    Returns (catalysts_list, meta) with source ('llm_gemini'|'llm_openai'|'mock'), cost, error.
    
    Provider order:
      1. LLM_ENABLED=false → mock data, $0
      2. Try gemini-2.5-flash with Google Search grounding (cheap, real facts)
      3. Fallback to gpt-4o (no search but stronger reasoning)
    """
    meta = {"source": None, "cost_usd": 0.0, "error": None}

    if not LLM_ENABLED:
        meta["source"] = "mock"
        return _mock_catalysts(ticker, company_name), meta

    if not _check_budget():
        meta["error"] = f"Daily LLM budget exceeded (${DAILY_LLM_BUDGET_USD})"
        return [], meta

    # Try primary: Gemini with grounding
    gemini_returned_empty = False
    if LLM_PROVIDER == "gemini" and GOOGLE_API_KEY:
        try:
            catalysts = _call_gemini_extract(ticker, company_name)
            _record_spend(COST_PER_GEMINI_CALL_USD)
            meta["source"] = "llm_gemini"
            meta["cost_usd"] = COST_PER_GEMINI_CALL_USD
            if catalysts:
                return catalysts, meta
            # Gemini returned empty list (silent JSON parse failure or content filter).
            # Fall through to OpenAI fallback rather than accept zero results.
            gemini_returned_empty = True
            meta["error"] = "gemini_returned_empty"
            logger.info(f"Gemini returned 0 catalysts for {ticker}, trying OpenAI fallback")
        except Exception as e:
            logger.warning(f"Gemini extract failed for {ticker}, trying OpenAI: {e}")
            meta["error"] = f"gemini_failed: {type(e).__name__}"
            # fall through to OpenAI

    # Fallback: OpenAI gpt-4o
    if not OPENAI_API_KEY:
        meta["error"] = (meta.get("error") or "") + "; OPENAI_API_KEY not set"
        return [], meta
    try:
        catalysts = _call_openai_extract(ticker, company_name)
        _record_spend(COST_PER_OPENAI_CALL_USD)
        # If we already recorded gemini cost, append rather than replace cost
        meta["cost_usd"] = round(meta.get("cost_usd", 0.0) + COST_PER_OPENAI_CALL_USD, 4)
        meta["source"] = "llm_openai_after_gemini_empty" if gemini_returned_empty else "llm_openai"
        return catalysts, meta
    except Exception as e:
        meta["error"] = (meta.get("error") or "") + f"; openai_failed: {type(e).__name__}: {e}"
        logger.exception(f"OpenAI extract failed for {ticker}")
        return [], meta


def _mock_catalysts(ticker: str, company_name: str) -> List[Dict]:
    """Mock catalysts to verify the plumbing works without LLM cost."""
    today = date.today()
    return [
        {
            "ticker": ticker,
            "company_name": company_name,
            "catalyst_type": "FDA Decision",
            "catalyst_date": (today + timedelta(days=90)).isoformat(),
            "date_precision": "exact",
            "description": f"[MOCK] Hypothetical FDA decision for {company_name} lead drug",
            "drug_name": "MOCK-001",
            "canonical_drug_name": "mock-drug-001",
            "indication": "Mock indication",
            "phase": "BLA",
            "source": "mock",
            "source_url": None,
            "confidence_score": 0.5,
        },
        {
            "ticker": ticker,
            "company_name": company_name,
            "catalyst_type": "Phase 3 Readout",
            "catalyst_date": (today + timedelta(days=120)).isoformat(),
            "date_precision": "quarter",
            "description": f"[MOCK] Hypothetical Phase 3 readout for {company_name}",
            "drug_name": "MOCK-002",
            "canonical_drug_name": "mock-drug-002",
            "indication": "Mock indication 2",
            "phase": "Phase 3",
            "source": "mock",
            "source_url": None,
            "confidence_score": 0.5,
        },
    ]


def _build_extraction_prompt(ticker: str, company_name: str) -> str:
    """Shared prompt for both Gemini and OpenAI."""
    today = date.today()
    six_months = today + timedelta(days=183)
    return f"""You are a biotech catalyst analyst. Extract REAL upcoming non-earnings catalysts for {ticker} ({company_name}) expected between {today.isoformat()} and {six_months.isoformat()}.

CRITICAL RULES:
- ONLY include catalysts you have factual evidence for (recent press releases, SEC 8-K filings, FDA calendar, ClinicalTrials.gov)
- DO NOT make up drug names, dates, or indications. If you don't know specifics, do not include the catalyst.
- Use EXACT drug brand/code names (e.g. "ABBV-001", "Humira", "Skyrizi") — never placeholders like "Drug X" or "Drug ABC"
- Indications must be specific medical conditions (e.g. "metastatic melanoma", "rheumatoid arthritis"), not "various"
- If you have NO high-confidence catalyst data for this ticker, return an empty array — do not invent

Catalyst types to find: FDA Decision, AdComm, Phase 1/2/3 Readout, Clinical Trial, Partnership, BLA submission, NDA submission.
EXCLUDE: earnings reports, conferences, investor days.

For each catalyst found:
- catalyst_type: one of the types above
- catalyst_date: ISO date YYYY-MM-DD (estimate quarter if uncertain)
- date_precision: "exact" | "quarter" | "half" | "year"
- description: 1-sentence FACTUAL description with specifics
- drug_name: real drug or program name (no placeholders)
- indication: specific disease/condition
- phase: "Phase 1" | "Phase 2" | "Phase 3" | "BLA" | "NDA" | null
- confidence_score: 0.0-1.0 — be honest about your certainty

Return ONLY a JSON object: {{"catalysts": [...]}}. If no high-confidence catalysts known, return {{"catalysts": []}}.
"""


def _call_openai_extract(ticker: str, company_name: str) -> List[Dict]:
    """Call OpenAI to extract upcoming catalysts. Returns parsed list of dicts."""
    prompt = _build_extraction_prompt(ticker, company_name)
    _ = """OLD_INLINE_PROMPT_REPLACED

"""

    body = {
        "model": LLM_MODEL_OPENAI,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers, timeout=45)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    
    # Parse JSON — model may return {"catalysts": [...]} or [...]
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        for k in ("catalysts", "results", "data", "items"):
            if k in parsed and isinstance(parsed[k], list):
                parsed = parsed[k]
                break
        else:
            parsed = []
    if not isinstance(parsed, list):
        parsed = []
    
    # Normalize each entry, attach ticker info
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        out.append({
            "ticker": ticker,
            "company_name": company_name,
            "catalyst_type": item.get("catalyst_type", "Unknown"),
            "catalyst_date": item.get("catalyst_date"),
            "date_precision": item.get("date_precision", "exact"),
            "description": item.get("description", ""),
            "drug_name": item.get("drug_name"),
            "canonical_drug_name": _canonicalize_drug(item.get("drug_name")),
            "indication": item.get("indication"),
            "phase": item.get("phase"),
            "source": "llm_openai",
            "source_url": None,
            "confidence_score": item.get("confidence_score", 0.6),
        })
    return out


def _extract_first_json_object(text: str) -> str:
    """
    Pull the first complete JSON object out of a possibly-noisy LLM response.
    Handles markdown fences, duplicated outputs, and trailing prose.
    """
    if not text:
        return ""
    
    # Strip markdown fences (```json ... ``` or ``` ... ```)
    if "```" in text:
        # Try to find content between first ``` and matching close
        start = text.find("```")
        # Skip the language specifier line
        nl = text.find("\n", start)
        if nl > 0:
            inner_start = nl + 1
            close = text.find("```", inner_start)
            if close > 0:
                text = text[inner_start:close]
            else:
                text = text[inner_start:]
    
    # Find first { and matching close brace via depth counter
    text = text.strip()
    if not text.startswith("{") and not text.startswith("["):
        # find first { or [
        b = text.find("{")
        s = text.find("[")
        first = min(p for p in [b, s] if p >= 0) if (b >= 0 or s >= 0) else -1
        if first < 0:
            return ""
        text = text[first:]
    
    # Walk depth — first complete object/array wins
    open_ch = text[0]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[: i + 1]
    return text  # incomplete, return what we have


def _call_gemini_extract(ticker: str, company_name: str) -> List[Dict]:
    """Call Gemini 2.5 Flash with Google Search grounding for real-time biotech facts."""
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=GOOGLE_API_KEY)
    prompt = _build_extraction_prompt(ticker, company_name)
    
    config = types.GenerateContentConfig(
        max_output_tokens=8000,         # generous so JSON isn't truncated mid-output
        temperature=0.1,                # low for factual extraction
    )
    if GEMINI_GROUNDING:
        # Enable Google Search as a tool — model decides when to use it
        config.tools = [types.Tool(google_search=types.GoogleSearch())]
    
    response = client.models.generate_content(
        model=LLM_MODEL_GEMINI,
        contents=prompt,
        config=config,
    )
    
    text = response.text or ""
    text = text.strip()
    
    # Gemini sometimes returns duplicated outputs concatenated. Find the FIRST complete JSON object.
    text = _extract_first_json_object(text)
    
    if not text:
        logger.warning(f"[gemini] {ticker} returned empty/unparseable text")
        return []
    
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"[gemini] {ticker} JSON decode failed: {e}; first 200 chars: {text[:200]}")
        return []
    if isinstance(parsed, dict):
        for k in ("catalysts", "results", "data", "items"):
            if k in parsed and isinstance(parsed[k], list):
                parsed = parsed[k]
                break
        else:
            parsed = []
    if not isinstance(parsed, list):
        parsed = []
    
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        # Filter out junk drug names from the model
        drug_name = item.get("drug_name", "")
        if drug_name and any(token in drug_name.lower() for token in [
            "drug x", "drug y", "drug z", "drug abc", "drug xyz",
            "drug 1", "drug 2", "compound x", "compound y", "compound z",
            "candidate x", "candidate y", "candidate z", "tbd", "n/a"
        ]):
            logger.warning(f"[gemini] {ticker} returned placeholder drug_name '{drug_name}', skipping")
            continue
        # Filter out generic indications
        indication = item.get("indication", "")
        if indication and indication.lower() in ("various", "various conditions", "tbd", "n/a", "unknown"):
            logger.warning(f"[gemini] {ticker} returned generic indication '{indication}', skipping")
            continue
        
        out.append({
            "ticker": ticker,
            "company_name": company_name,
            "catalyst_type": item.get("catalyst_type", "Unknown"),
            "catalyst_date": item.get("catalyst_date"),
            "date_precision": item.get("date_precision", "exact"),
            "description": item.get("description", ""),
            "drug_name": drug_name or None,
            "canonical_drug_name": _canonicalize_drug(drug_name),
            "indication": indication or None,
            "phase": item.get("phase"),
            "source": "llm_gemini",
            "source_url": None,
            "confidence_score": item.get("confidence_score", 0.7),
        })
    return out


def _canonicalize_drug(name: Optional[str]) -> Optional[str]:
    """Lowercase + strip + collapse whitespace."""
    if not name:
        return None
    return " ".join(name.lower().split())


# ────────────────────────────────────────────────────────────
# DB writer
# ────────────────────────────────────────────────────────────
def write_catalysts_to_db(catalysts: List[Dict], conn) -> Dict:
    """
    Upsert catalysts into catalyst_universe.
    Returns {"added": int, "updated": int, "skipped": int, "errors": [...]}
    """
    stats = {"added": 0, "updated": 0, "skipped": 0, "errors": []}
    
    with conn.cursor() as cur:
        for c in catalysts:
            try:
                raw_date = c.get("catalyst_date")
                if not raw_date:
                    stats["skipped"] += 1
                    continue
                
                # Normalize date format (handle '2026', '2026-Q3', '2026-08', etc.)
                normalized_date, normalized_precision = _normalize_catalyst_date(raw_date)
                if not normalized_date:
                    stats["errors"].append(f"{c.get('ticker','?')}/{c.get('catalyst_type','?')}: unparseable date '{raw_date}'")
                    stats["skipped"] += 1
                    continue
                c["catalyst_date"] = normalized_date
                # Override precision only if parsing detected a coarser one
                if normalized_precision != "exact":
                    c["date_precision"] = normalized_precision
                
                # Try INSERT, fall back to UPDATE on conflict
                cur.execute("""
                    INSERT INTO catalyst_universe
                        (ticker, company_name, catalyst_type, catalyst_date, date_precision,
                         description, drug_name, canonical_drug_name, indication, phase,
                         source, source_url, confidence_score, status, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active', NOW())
                    ON CONFLICT (ticker, catalyst_type, catalyst_date, drug_name)
                    DO UPDATE SET
                        company_name = EXCLUDED.company_name,
                        description = EXCLUDED.description,
                        indication = EXCLUDED.indication,
                        phase = EXCLUDED.phase,
                        confidence_score = EXCLUDED.confidence_score,
                        source = EXCLUDED.source,
                        last_updated = NOW()
                    RETURNING (xmax = 0) AS inserted
                """, (
                    c["ticker"], c.get("company_name"), c["catalyst_type"],
                    c["catalyst_date"], c.get("date_precision", "exact"),
                    c.get("description"), c.get("drug_name"), c.get("canonical_drug_name"),
                    c.get("indication"), c.get("phase"),
                    c.get("source", "unknown"), c.get("source_url"),
                    c.get("confidence_score"),
                ))
                inserted = cur.fetchone()[0]
                if inserted:
                    stats["added"] += 1
                else:
                    stats["updated"] += 1
            except Exception as e:
                stats["errors"].append(f"{c.get('ticker')}/{c.get('catalyst_type')}: {e}")
        conn.commit()
    return stats


# ────────────────────────────────────────────────────────────
# Top-level orchestration
# ────────────────────────────────────────────────────────────
def run_universe_seed(max_tickers: Optional[int] = None) -> Dict:
    """
    Main entry point. Run from /admin/universe/v2-seed endpoint or cron.
    Returns a summary dict with counts + cost incurred + errors.
    """
    import psycopg2
    
    started = datetime.utcnow()
    db_url = os.getenv("DATABASE_URL", "")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    conn = psycopg2.connect(db_url)
    
    # Log cron run start
    cron_run_id = None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO cron_runs (job_name, status) VALUES ('universe_seed_v2', 'running')
                RETURNING id
            """)
            cron_run_id = cur.fetchone()[0]
            conn.commit()
    except Exception as e:
        logger.warning(f"cron_runs insert failed: {e}")
    
    universe = get_seed_universe()
    cap = max_tickers or MAX_TICKERS_PER_RUN
    universe = universe[:cap]
    
    logger.info(f"[universe_seed] LLM_ENABLED={LLM_ENABLED} provider={LLM_PROVIDER} processing {len(universe)} tickers (cap={cap})")
    
    total_added = 0
    total_updated = 0
    total_errors = []
    total_cost = 0.0
    catalysts_per_source = {"mock": 0, "llm_gemini": 0, "llm_openai": 0, "error": 0}
    
    for entry in universe:
        ticker = entry["ticker"]
        company = entry["company_name"]
        try:
            catalysts, meta = extract_catalysts_for_ticker(ticker, company)
            total_cost += meta.get("cost_usd", 0.0)
            
            if meta.get("source"):
                catalysts_per_source[meta["source"]] = catalysts_per_source.get(meta["source"], 0) + len(catalysts)
            
            if meta.get("error"):
                total_errors.append(f"{ticker}: {meta['error']}")
                catalysts_per_source["error"] += 1
                continue
            
            if catalysts:
                stats = write_catalysts_to_db(catalysts, conn)
                total_added += stats["added"]
                total_updated += stats["updated"]
                if stats["errors"]:
                    total_errors.extend(stats["errors"])
        except Exception as e:
            total_errors.append(f"{ticker}: orchestrator error: {e}")
            logger.exception(f"orchestrator error for {ticker}")
    
    completed = datetime.utcnow()
    duration_s = (completed - started).total_seconds()
    
    summary = {
        "job": "universe_seed_v2",
        "started_at": started.isoformat() + "Z",
        "completed_at": completed.isoformat() + "Z",
        "duration_seconds": round(duration_s, 1),
        "tickers_processed": len(universe),
        "catalysts_added": total_added,
        "catalysts_updated": total_updated,
        "catalysts_per_source": catalysts_per_source,
        "total_llm_cost_usd": round(total_cost, 4),
        "daily_spend_after": get_daily_spend(),
        "llm_enabled": LLM_ENABLED,
        "errors": total_errors[:20],  # Cap error list
        "error_count": len(total_errors),
    }
    
    # Update cron run
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE cron_runs SET
                    completed_at = NOW(),
                    status = %s,
                    records_processed = %s,
                    records_added = %s,
                    records_updated = %s,
                    errors = %s,
                    log = %s
                WHERE id = %s
            """, (
                "success" if not total_errors else ("partial" if total_added > 0 else "failed"),
                len(universe), total_added, total_updated,
                json.dumps(total_errors[:50]) if total_errors else None,
                json.dumps(summary),
                cron_run_id,
            ))
            conn.commit()
    except Exception as e:
        logger.warning(f"cron_runs update failed: {e}")
    
    conn.close()
    return summary
