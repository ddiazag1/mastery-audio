"""
Generate a single ABNS oral board viva-style audio script from a concept.
Usage: python generate_concept.py <concept_name> [--module neuro_boards]
"""

import anthropic
import json
import sys
import os
import re
from pathlib import Path

# Supabase config (read-only anon key)
SUPABASE_URL = "https://vxtaghytxnycegztmrml.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZ4dGFnaHl0eG55Y2VnenRtcm1sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzEwOTM2MzIsImV4cCI6MjA4NjY2OTYzMn0.3E0IFwp6LIcs7tChpj28jgaerf76zCB-oCbjRJkAtDU"

MODULE_PROMPTS = {
    "neuro_boards": """You are a senior ABNS oral board examiner and neurosurgery professor creating a comprehensive audio teaching script. Your goal is to produce content that, when listened to repeatedly, will prepare a candidate to PASS the ABNS oral board examination on this topic.

TARGET: 3,000-3,500 words. This will be read aloud at ~150 wpm for a 20-25 minute audio segment.

TOPIC: {name}
CATEGORY: {category} > {subcategory}
DIFFICULTY: {difficulty}/5
SEED CONTENT: {description}

Write the script in this exact structure:

## SECTION 1: FOUNDATION (3-4 minutes)
Open with: "Let's master [topic] for the oral boards."
- Core anatomy/pathophysiology relevant to this topic
- Classification systems with ALL grades/types spelled out (don't just name them — define each grade)
- Epidemiology and natural history
- Key imaging findings described in detail
This section should be dense with facts. An examiner expects you to rattle these off fluently.

## SECTION 2: CLINICAL CASE (2-3 minutes)
Present a realistic case that an examiner would use:
"You're consulted on a [age]-year-old [sex] who presents with..."
- Include relevant history, exam findings, vitals
- Describe imaging findings as they would appear
- Make it a bread-and-butter presentation, not a zebra

## SECTION 3: EXAMINER INTERROGATION (8-12 minutes)
This is the core. Simulate an actual viva examination.
Write it as a back-and-forth dialogue:

EXAMINER: "What is your differential diagnosis?"
YOUR ANSWER: [Provide the model answer — what the examiner wants to hear, in the order they want to hear it]

EXAMINER: "How would you manage this patient?"
YOUR ANSWER: [Step-by-step management]

Continue through ALL major decision points:
- Initial workup and management
- Surgical indications (be specific: cite size cutoffs, score thresholds, clinical criteria)
- Surgical approach selection with reasoning
- Intraoperative decision-making and technique highlights
- "What if" complications during surgery
- Postoperative management
- "The patient deteriorates" — complication management

Include at least 2-3 TRAP QUESTIONS that examiners commonly use to fail candidates. Mark these clearly:
EXAMINER [TRAP]: "Would you..."
YOUR ANSWER: [Explain why this is a trap and what the correct answer is]

## SECTION 4: EVIDENCE AND GUIDELINES (2-3 minutes)
- Landmark trials by name with key findings (e.g., "The SPORT trial showed...")
- Current guidelines (BTF, CNS/AANS) with specific recommendations
- Level of evidence where relevant
- Any recent paradigm shifts

## SECTION 5: BOARD PEARLS (2-3 minutes)
Close with rapid-fire high-yield points:
- "Remember: [fact]"
- "Never say: [common mistake]. Instead say: [correct framing]"
- "If they ask about [X], they're testing whether you know [Y]"
- "The three things that will fail you on this topic: [1, 2, 3]"
End with: "That covers [topic]. You're ready for this one."

CRITICAL QUALITY RULES:
1. Every numerical value must be specific (don't say "large" — say ">3cm" or ">10mm")
2. Every classification must include ALL grades/types with definitions
3. Every management decision must include the INDICATION and the ALTERNATIVE
4. Use the exact terminology an examiner expects ("maximal safe resection" not "remove as much as possible")
5. Surgical approaches must be named precisely (e.g., "far-lateral suboccipital" not "posterior approach")
6. Drug doses should be included where they are standard-of-care
7. Don't hedge — give definitive answers as a confident candidate would
8. If there is genuine controversy, state both sides and then state what you would say on boards
9. Assume the listener is a neurosurgery-trained physician — do not explain basic anatomy unless it's specifically relevant to the decision-making
10. Write for AUDIO — no bullet points, no tables, no abbreviations without first spelling them out. Use natural spoken language.
11. Do NOT use markdown formatting, asterisks, or headers in the output — write pure prose with clear verbal transitions like "Moving to section two" or "Now let's talk about the evidence"
""",

    "ent_business": """You are a healthcare business consultant and ENT practice management expert creating a comprehensive audio teaching script for an ENT surgeon learning practice management.

TARGET: 3,000-3,500 words for 20-25 minute audio.

TOPIC: {name}
CATEGORY: {category} > {subcategory}
DIFFICULTY: {difficulty}/5
SEED CONTENT: {description}

Structure:
1. CONCEPT INTRODUCTION (3-4 min): What this is and why it matters for an ENT practice
2. DEEP DIVE (8-10 min): Full explanation with ENT-specific examples, real numbers, benchmarks
3. CASE SCENARIO (5-7 min): Walk through a realistic practice situation applying this concept
4. ACTION ITEMS (3-4 min): Specific steps to implement or evaluate in your own practice
5. KEY TAKEAWAYS (2 min): Summary of must-remember points

Use specific ENT CPT codes, MGMA benchmarks, and real-world dollar amounts. Write for audio — natural spoken language, no bullet points or markdown formatting.
""",

    "leadership": """You are a physician leadership coach and organizational psychologist creating a comprehensive audio teaching script for a surgeon developing leadership capabilities.

TARGET: 3,000-3,500 words for 20-25 minute audio.

TOPIC: {name}
CATEGORY: {category} > {subcategory}
DIFFICULTY: {difficulty}/5
SEED CONTENT: {description}

Structure:
1. FRAMEWORK INTRODUCTION (3-4 min): Origin and core principles
2. DEEP EXPLANATION (8-10 min): Each component explained with examples
3. PHYSICIAN APPLICATION (5-7 min): How this applies specifically in surgical/clinical settings with concrete scenarios
4. PRACTICE EXERCISE (3-4 min): A mental exercise or reflection prompt the listener can apply this week
5. SYNTHESIS (2 min): How this connects to being a better surgeon-leader

Reference original authors and research. Write for audio — natural spoken language, no bullet points or markdown formatting.
""",

    "capital_deployment": """You are a physician financial strategist and acquisition advisor creating a comprehensive audio teaching script for a high-income surgeon learning capital deployment.

TARGET: 3,000-3,500 words for 20-25 minute audio.

TOPIC: {name}
CATEGORY: {category} > {subcategory}
DIFFICULTY: {difficulty}/5
SEED CONTENT: {description}

Structure:
1. CONCEPT OVERVIEW (3-4 min): What this is and why it matters for a physician investor
2. MECHANICS (8-10 min): Detailed explanation with real numbers, formulas, worked examples
3. DEAL WALKTHROUGH (5-7 min): Step through a realistic scenario with specific dollar amounts
4. RISK ANALYSIS (3-4 min): What can go wrong, red flags, how to protect yourself
5. ACTION STEPS (2 min): What to do this month to move forward on this

Use real numbers throughout ($X revenue, Y% returns, $Z costs). Write for audio — natural spoken language, no bullet points or markdown formatting.
"""
}


def fetch_concepts(module: str):
    """Fetch all concepts for a module from Supabase."""
    import urllib.request
    url = f"{SUPABASE_URL}/rest/v1/sm_concepts?module=eq.{module}&order=sort_order"
    req = urllib.request.Request(url, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    })
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def generate_script(concept: dict, module: str) -> str:
    """Generate a full audio script for a concept using Claude."""
    client = anthropic.Anthropic()

    prompt_template = MODULE_PROMPTS[module]
    prompt = prompt_template.format(
        name=concept["name"],
        category=concept["category"],
        subcategory=concept["subcategory"],
        difficulty=concept["difficulty"],
        description=concept["description"],
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514" if False else "claude-3-5-sonnet-20241022",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


def slugify(name: str) -> str:
    """Convert concept name to filename slug."""
    s = name.lower().strip()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s_]+', '-', s)
    s = re.sub(r'-+', '-', s)
    return s


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate audio script for a concept")
    parser.add_argument("concept_name", help="Name of the concept (or 'all' for all concepts)")
    parser.add_argument("--module", default="neuro_boards", help="Module name")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path(__file__).parent.parent / "content" / args.module.replace("_", "-"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching concepts for module: {args.module}")
    concepts = fetch_concepts(args.module)
    print(f"Found {len(concepts)} concepts")

    if args.concept_name.lower() == "all":
        targets = concepts
    else:
        targets = [c for c in concepts if args.concept_name.lower() in c["name"].lower()]

    if not targets:
        print(f"No concepts found matching '{args.concept_name}'")
        print("Available concepts:")
        for c in concepts:
            print(f"  - {c['name']}")
        sys.exit(1)

    for concept in targets:
        slug = slugify(concept["name"])
        outfile = os.path.join(output_dir, f"{slug}.md")

        if os.path.exists(outfile):
            print(f"SKIP (exists): {concept['name']} -> {outfile}")
            continue

        print(f"GENERATING: {concept['name']}...")
        script = generate_script(concept, args.module)

        # Write with metadata header
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(f"---\n")
            f.write(f"title: \"{concept['name']}\"\n")
            f.write(f"module: {args.module}\n")
            f.write(f"category: \"{concept['category']}\"\n")
            f.write(f"subcategory: \"{concept['subcategory']}\"\n")
            f.write(f"difficulty: {concept['difficulty']}\n")
            f.write(f"---\n\n")
            f.write(script)

        word_count = len(script.split())
        est_minutes = word_count / 150
        print(f"  -> {outfile} ({word_count} words, ~{est_minutes:.1f} min audio)")

    print("\nDone.")


if __name__ == "__main__":
    main()
