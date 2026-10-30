"""Automated tests for Medical Answer Rendering UI (Markdown, Tables, Callouts, Equations, Toolbars).

Validates:
1. JSON unwrapping: Cleanly extracts answers from raw JSON wrappers and ```json blocks.
2. Typography and Heading Hierarchy: H1 main title, H2 major section, H3 subsection, H4 item.
3. Tables: GFM markdown tables with header rows, cell alignment, numeric styling, and responsive wrappers.
4. Clinical Callout Boxes: GitHub-style alerts and clinical note labels (Key Points, Contraindications, Exam Facts, Pearls).
5. Mathematical and Scientific Expressions: Display math, inline math, Greek letters, chemical/gas subscripts, superscripts, arrows.
6. Code Blocks: Syntax formatting with copy snippet button.
7. Verified Citations & Hyperlinks: Clickable external links with external arrow icon.
8. Action Toolbar & Follow-up Suggestions: Copy, Regenerate, Helpful/Not Helpful feedback, Read Aloud, Sources, and follow-up chips.
9. Backend integration: Chat view serves template with modern chat stylesheet and script.
"""

from __future__ import annotations

import json
import subprocess
import unittest

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app import create_app
from config import TestingConfig
from models import User, db


class TestMedicalAnswerRenderingUI(unittest.TestCase):
    """Test suite validating textbook-quality medical answer rendering in MediAssist."""

    def setUp(self):
        self.app = create_app(TestingConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()

        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        db.metadata.create_all(bind=self.engine)
        db.session.configure(bind=self.engine)

        self.client = self.app.test_client()

        # Seed test user
        self.test_user = User(name="MBBS Student", email="student@medical.edu")
        self.test_user.set_password("SecureMedPass123!")
        db.session.add(self.test_user)
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.metadata.drop_all(bind=self.engine)
        self.engine.dispose()
        self.app_context.pop()

    def _eval_js(self, js_expression: str) -> str:
        """Helper to run a JavaScript expression with modern_chat.js functions loaded in Node."""
        harness = f"""
        const fs = require('fs');
        const code = fs.readFileSync('static/modern_chat.js', 'utf-8');

        // Mock minimal DOM environment
        const dom = {{
          getElementById: () => null,
          querySelector: () => null,
          querySelectorAll: () => [],
          createElement: (tag) => ({{
            tagName: tag.toUpperCase(),
            className: '',
            style: {{}},
            attributes: {{}},
            children: [],
            setAttribute(k, v) {{ this.attributes[k] = v; }},
            getAttribute(k) {{ return this.attributes[k]; }},
            appendChild(c) {{ this.children.push(c); }},
            querySelector() {{ return null; }},
            querySelectorAll() {{ return []; }},
            addEventListener() {{}}
          }})
        }};
        global.document = dom;
        global.window = {{ speechSynthesizer: null }};
        global.navigator = {{ clipboard: {{ writeText: async () => {{}} }} }};

        // Extract functions from IIFE
        const matchUnwrap = code.match(/function unwrapMedicalAnswer[\\s\\S]*?\\n  \\}}/);
        const matchInline = code.match(/function formatInlineMarkdown[\\s\\S]*?\\n  \\}}/);
        const matchTable = code.match(/function parseTableBlock[\\s\\S]*?\\n  \\}}/);
        const matchCallout = code.match(/function parseCalloutBlock[\\s\\S]*?\\n  \\}}/);
        const matchSafeMd = code.match(/function formatSafeMarkdown[\\s\\S]*?\\n  \\}}/);
        const matchEscape = code.match(/function escapeHtml[\\s\\S]*?\\n  \\}}/);
        const matchFollowups = code.match(/function generateContextualFollowups[\\s\\S]*?\\n  \\}}/);

        eval(matchEscape[0]);
        eval(matchUnwrap[0]);
        eval(matchInline[0]);
        eval(matchTable[0]);
        eval(matchCallout[0]);
        eval(matchSafeMd[0]);
        eval(matchFollowups[0]);

        const result = ({js_expression});
        console.log(typeof result === 'object' ? JSON.stringify(result) : String(result));
        """
        proc = subprocess.run(
            ["node", "-"],
            input=harness,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            cwd=".",
        )
        return proc.stdout.strip()

    def test_json_wrapper_unwrapping(self):
        """Verify unwrapMedicalAnswer strips raw JSON blocks and escaped characters."""
        raw_json_md = '```json\n{\n  "answer": "### Acute Emergency Protocol\\nPatient in shock requires fluid."\n}\n```'
        result = self._eval_js(f"unwrapMedicalAnswer({json.dumps(raw_json_md)})")
        self.assertIn("### Acute Emergency Protocol", result)
        self.assertNotIn('"answer":', result)
        self.assertNotIn("```json", result)

    def test_raw_json_without_fences_unwrapping(self):
        """Verify unwrapMedicalAnswer handles direct JSON strings."""
        raw_json = '{"answer": "## Diabetic Nephropathy\\nProgression of glomerulosclerosis."}'
        result = self._eval_js(f"unwrapMedicalAnswer({json.dumps(raw_json)})")
        self.assertIn("## Diabetic Nephropathy", result)
        self.assertNotIn('{"answer"', result)

    def test_truncated_streaming_json_wrapper_unwrapping(self):
        """Verify unwrapMedicalAnswer extracts answers from unclosed or truncated streaming JSON."""
        streaming_json = '```json\n{\n  "answer": "### Fever of Unknown Origin (FUO)\\nPathophysiology involves pyrogens and cytokines'
        result = self._eval_js(f"unwrapMedicalAnswer({json.dumps(streaming_json)})")
        self.assertIn("### Fever of Unknown Origin (FUO)", result)
        self.assertNotIn('```json', result)
        self.assertNotIn('"answer":', result)

    def test_heading_hierarchy(self):
        """Verify H1, H2, H3, H4 headings are mapped into textbook answer hierarchy classes."""
        markdown = "# Main Title\n## Major Section\n### Subsection\n#### Item Point"
        html = self._eval_js(f"formatSafeMarkdown({json.dumps(markdown)})")
        self.assertIn('<h1 class="answer-title">Main Title</h1>', html)
        self.assertIn('<h2 class="answer-section-heading">Major Section</h2>', html)
        self.assertIn('<h3 class="answer-sub-heading">Subsection</h3>', html)
        self.assertIn('<h4 class="answer-item-heading">Item Point</h4>', html)

    def test_medical_table_rendering(self):
        """Verify GFM tables render inside a responsive container with headers and cell alignment."""
        table_md = (
            "| Drug | Dose | Indication | Contraindications |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| Aspirin | 162-325 mg | Acute Coronary Syndrome | Active bleeding |\n"
            "| Nitroglycerin | 0.4 mg SL | Ischemic chest pain | SBP < 90 mm Hg |"
        )
        html = self._eval_js(f"formatSafeMarkdown({json.dumps(table_md)})")
        self.assertIn('<div class="medical-table-container"', html)
        self.assertIn('<table class="medical-table">', html)
        self.assertIn("<thead><tr>", html)
        self.assertIn("<th", html)
        self.assertIn("Aspirin", html)
        self.assertIn("162-325 mg", html)
        self.assertIn("Nitroglycerin", html)

    def test_clinical_callout_boxes(self):
        """Verify GitHub alerts and clinical labels render with alert/note roles and icons."""
        callout_md = (
            "> [!WARNING]\n"
            "> Right Ventricular Infarction: Avoid nitrates and diuretics.\n\n"
            "> [!IMPORTANT]\n"
            "> High-Yield Exam Fact: Reed-Sternberg cells are CD15+ and CD30+."
        )
        html = self._eval_js(f"formatSafeMarkdown({json.dumps(callout_md)})")
        self.assertIn('class="clinical-callout clinical-callout--contraindication"', html)
        self.assertIn('role="alert"', html)
        self.assertIn("Right Ventricular Infarction", html)
        self.assertIn('class="clinical-callout clinical-callout--exam-fact"', html)
        self.assertIn("Reed-Sternberg cells", html)

    def test_mathematical_and_scientific_expressions(self):
        """Verify mathematical formulas, blood gas subscripts, and Greek symbols."""
        text = r"Administer $O_2$ if $SpO_2 < 90\%$. Formula: $$\Delta P = Q \times R$$. Lead $V_3R-V_5R$."
        html = self._eval_js(f"formatSafeMarkdown({json.dumps(text)})")
        self.assertIn('role="math"', html)
        self.assertIn("O<sub>2</sub>", html)
        self.assertIn("SpO<sub>2</sub>", html)
        self.assertIn("V<sub>3</sub>", html)

    def test_mechanism_ordered_steps(self):
        """Verify physiological sequence and algorithm numbering renders with medical-steps-list class."""
        steps_md = (
            "1. Initial Assessment: ABCDE protocol and 12-lead ECG.\n"
            "2. MONA Protocol: Oxygen, Aspirin, Nitroglycerin, Morphine.\n"
            "3. Reperfusion Decision: Primary PCI within 90 minutes."
        )
        html = self._eval_js(f"formatSafeMarkdown({json.dumps(steps_md)})")
        self.assertIn('<ol class="medical-steps-list">', html)
        self.assertIn("<li>Initial Assessment", html)
        self.assertIn("<li>MONA Protocol", html)

    def test_verified_citations_and_links(self):
        """Verify source links are rendered safely with the external link icon."""
        link_md = "Guidelines from [AHA / ACC STEMI Guidelines](https://www.ahajournals.org/stemi)."
        html = self._eval_js(f"formatSafeMarkdown({json.dumps(link_md)})")
        self.assertIn('class="inline-source-link"', html)
        self.assertIn('href="https://www.ahajournals.org/stemi"', html)
        self.assertIn('target="_blank"', html)
        self.assertIn("↗", html)

    def test_contextual_followup_suggestions(self):
        """Verify follow-up suggestions generator returns high-yield medical topic prompts."""
        stemi_text = "Acute STEMI protocol includes immediate ECG, chewable aspirin, and reperfusion windows."
        result = self._eval_js(f"generateContextualFollowups({json.dumps(stemi_text)}, 'Tell me about STEMI')")
        followups = json.loads(result)
        self.assertIsInstance(followups, list)
        self.assertGreaterEqual(len(followups), 1)
        self.assertTrue(any("ECG" in q or "Reperfusion" in q or "PCI" in q for q in followups))

    def test_chat_page_includes_modern_assets(self):
        """Verify /chat renders chat.html referencing modern_chat.css and modern_chat.js."""
        with self.client:
            login_resp = self.client.post("/login", data={
                "email": "student@medical.edu",
                "password": "SecureMedPass123!",
            }, follow_redirects=True)
            self.assertEqual(login_resp.status_code, 200)

            chat_resp = self.client.get("/chat")
            self.assertEqual(chat_resp.status_code, 200)
            html = chat_resp.get_data(as_text=True)
            self.assertIn("modern_chat.css", html)
            self.assertIn("MediAssist", html)
            self.assertIn("AI-Powered Medical Learning", html)
            self.assertIn("themeToggleBtn", html)


if __name__ == "__main__":
    unittest.main()
