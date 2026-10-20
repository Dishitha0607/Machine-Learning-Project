import re

with open('app/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_header = """# ---------------------------------------------------------------------------
# CLEAN TOP HEADER
# ---------------------------------------------------------------------------
st.markdown(\"\"\"
<div class="academic-header">
    <div class="header-title">Machine Learning <span>Capstone Dashboard</span></div>
    <div class="header-subtitle">
        B.Tech. Computer Science & Engineering (23CSE301) \u2014 Unified Performance Evaluation & Interactive Inference System.<br>
        All 22 algorithms evaluated across Regression, Classification, and Clustering with zero discrepancy to notebook outputs.
    </div>
    <div class="header-chips">
        <div class="header-chip"><span class="chip-dot green"></span> Course: 23CSE301</div>
        <div class="header-chip"><span class="chip-dot"></span> Evaluated Models: 22 / 22</div>
        <div class="header-chip"><span class="chip-dot"></span> Split: 80% Train / 20% Test</div>
        <div class="header-chip"><span class="chip-dot green"></span> Notebook Output Match: 100%</div>
        <div class="header-chip"><span class="chip-dot amber"></span> Bonus: Interactive GUI &amp; Deployment Ready</div>
    </div>
</div>
\"\"\", unsafe_allow_html=True)"""

new_header = """# ---------------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------------
st.markdown(\"\"\"
<div class="academic-header">
    <div class="header-title">InsightIQ <span>ML Analytics Platform</span></div>
    <div class="header-subtitle">
        Unified intelligent analytics for Regression, Classification &amp; Clustering.<br>
        22 machine learning models trained, evaluated, and available for live inference in one place.
    </div>
    <div class="header-chips">
        <div class="header-chip"><span class="chip-dot green"></span> 22 Models Active</div>
        <div class="header-chip"><span class="chip-dot"></span> 80 / 20 Train-Test Split</div>
        <div class="header-chip"><span class="chip-dot"></span> 3 Problem Domains</div>
        <div class="header-chip"><span class="chip-dot green"></span> Live Predictions Enabled</div>
        <div class="header-chip"><span class="chip-dot amber"></span> Interactive Visualizations</div>
    </div>
</div>
\"\"\", unsafe_allow_html=True)"""

if old_header in content:
    content = content.replace(old_header, new_header, 1)
    with open('app/main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Header replaced successfully')
else:
    print('NOT FOUND - searching for partial match...')
    idx = content.find('CLEAN TOP HEADER')
    if idx >= 0:
        print(repr(content[idx:idx+600]))
