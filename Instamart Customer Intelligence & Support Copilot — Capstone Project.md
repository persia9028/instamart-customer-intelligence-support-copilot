Q: 1
Instamart Customer Intelligence & Support Copilot — Capstone Project
Swiggy Instamart delivers groceries, dairy, snacks, personal-care items and small electronics in six metro cities within minutes. As the order base grows, three problems keep surfacing in the weekly leadership review: nobody trusts the raw order exports (duplicate rows, mixed spellings, impossible values), the retention team cannot tell which customers are about to stop ordering, and the support desk is drowning in thousands of free-text tickets that mix refunds, fraud reports, app crashes and late deliveries. The Head of Customer Experience has asked a small data and AI team to build one internal tool that fixes all three: a Customer Intelligence & Support Copilot.

You are that team. Over 21 days you will build the Copilot as one repository, in four connected Parts. Part A turns the messy exports into a trusted analytical database. Part B uses that database to segment customers and to predict who is likely to lapse. Part C reads support tickets, protects customer privacy, and routes each ticket using the segments from Part B. Part D builds a policy question-answering assistant and a refund agent that queries the database from Part A and reuses the policy retrieval built in the same Part. Every Part consumes what the previous Parts produced, so earlier mistakes travel forward — and the numeric checkpoints in this brief let you catch them early.

All data in this project is synthetic and fictional, produced by a script given below. The brand name is used only to frame the scenario; nothing here is real Instamart data, policy or pricing. All monetary values are in Indian Rupees (INR).

Total marks: 100 · One deliverable: a single public GitHub repository containing the complete Copilot (data pipeline, SQL, models, support-triage pipeline, policy assistant, refund agent) and one README.md that documents it.

Part	Title	Marks
A	Data Foundation: Clean, Store, Query	22
B	Customer Segments and Lapse Prediction	28
C	Support Ticket Triage with Guardrails	22
D	Policy RAG Assistant and Refund Agent	28
Submission Guidelines (read first)
One submission for the whole project: a single public GitHub repository link. There are no per-Part submissions. The repository must be accessible without logging in.
One branch, named main. All of your work must live on the main branch, and the link you submit must point to that exact branch (for example https://github.com/<user>/<repo>/tree/main). A link to any other branch, or a link that does not clearly resolve to a branch, will not be graded.
Text only. Code is submitted through the repository and everything else as text inside README.md. No images, screenshots, PDFs, slide decks, presentations, video or audio are required or accepted. Charts are optional, are not graded, and must not be committed; describe every finding in numbers and words.
No paid or account-gated service is needed anywhere in this project. Every language-model call goes through the supplied offline file mock_llm.py (given verbatim below), which needs no API key, account, network connection or money. All libraries are free and open source. You may optionally experiment with a hosted model on your own, but nothing in this brief or its grading depends on it; if you do, keep any key in a .env file that is listed in .gitignore and never commit it.
Allowed libraries: the Python standard library (including sqlite3, re, json, random, time, math, csv, os), pandas, numpy, scikit-learn and joblib. Python 3.11 or newer; SQLite 3.25 or newer (window functions) — the version bundled with current Python releases is sufficient.
Reproducibility is graded. A grader will clone your repository into a clean folder, create a virtual environment, run the commands listed in your README's Reproduce section in order, and compare the results with the checkpoints in this brief. Fixed random seeds are specified wherever randomness appears.
Originality and academic integrity. The two supplied files generate_data.py and mock_llm.py must be used unmodified; everything else must be your own work. You may consult documentation. Copying another student's code is a violation of academic integrity and will be treated as such.
Deadline: 21 days from the day the brief is released, as confirmed by the programme administration. A suggested (non-binding) pacing: Part A days 1–5, Part B days 6–11, Part C days 12–15, Part D days 16–21.
Reference checkpoints. Numbers marked reference below were produced by running the supplied scripts with Python 3, pandas, numpy and scikit-learn. Counts and sums are exact. Values from scikit-learn estimators (K-means, PCA, forests) carry the stated tolerance.
The Supplied Files
Create these two files at the repository root, copy them exactly, and commit them unmodified.

generate_data.py
Run python generate_data.py once. It writes data/raw/customers.csv, products.csv, orders.csv, tickets.csv, policy_eval.csv, agent_scenarios.csv and five policy text files in data/policies/. Its last line of output must be exactly:

generated: customers=1200 products=40 orders_raw=13084 tickets=164

"""generate_data.py -- deterministic synthetic data for the Instamart Customer Intelligence capstone.
Run:  python generate_data.py        (writes ./data/raw/*.csv and ./data/policies/*.txt)
Uses only the Python standard library. All data is synthetic and fictional."""
import csv, math, os, random
from datetime import datetime, timedelta

rng = random.Random(42)
RAW, POL = os.path.join("data", "raw"), os.path.join("data", "policies")
os.makedirs(RAW, exist_ok=True); os.makedirs(POL, exist_ok=True)

CITIES = ["Chennai", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune"]
PRODUCTS = [  # (name, category, price_inr)
    ("Basmati Rice 1kg", "Grocery", 129), ("Toor Dal 1kg", "Grocery", 165), ("Atta 5kg", "Grocery", 245),
    ("Sunflower Oil 1L", "Grocery", 155), ("Sugar 1kg", "Grocery", 48), ("Salt 1kg", "Grocery", 24),
    ("Tea Powder 250g", "Grocery", 120), ("Poha 500g", "Grocery", 42),
    ("Toned Milk 500ml", "Dairy", 29), ("Curd 400g", "Dairy", 45), ("Paneer 200g", "Dairy", 95),
    ("Butter 100g", "Dairy", 58), ("Cheese Slices 200g", "Dairy", 130), ("Ghee 500ml", "Dairy", 340),
    ("Eggs 12 Pack", "Dairy", 84), ("Lassi 200ml", "Dairy", 25),
    ("Potato Chips 52g", "Snacks", 20), ("Namkeen 200g", "Snacks", 60), ("Biscuits 150g", "Snacks", 30),
    ("Dark Chocolate 80g", "Snacks", 150), ("Instant Noodles 4 Pack", "Snacks", 56), ("Popcorn 100g", "Snacks", 45),
    ("Peanut Butter 340g", "Snacks", 210), ("Cold Coffee Can 180ml", "Snacks", 90),
    ("Shampoo 340ml", "Personal Care", 280), ("Face Wash 100ml", "Personal Care", 190),
    ("Toothpaste 150g", "Personal Care", 98), ("Soap 4 Pack", "Personal Care", 150),
    ("Hand Wash 250ml", "Personal Care", 99), ("Body Lotion 400ml", "Personal Care", 350),
    ("Deodorant 150ml", "Personal Care", 220), ("Sunscreen 50g", "Personal Care", 399),
    ("Wireless Earbuds", "Electronics", 1499), ("Power Bank 10000mAh", "Electronics", 1299),
    ("USB-C Cable 1m", "Electronics", 299), ("Phone Charger 20W", "Electronics", 799),
    ("Smart Bulb", "Electronics", 599), ("Bluetooth Speaker", "Electronics", 1999),
    ("Phone Stand", "Electronics", 349), ("Memory Card 64GB", "Electronics", 549)]
CAT_IDX = {}
for i, (_, c, _) in enumerate(PRODUCTS):
    CAT_IDX.setdefault(c, []).append(i)

def write(name, header, rows):
    with open(os.path.join(RAW, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def poisson(lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1

def mangle_city(c):
    r = rng.random()
    return c.lower() if r < .20 else c.upper() if r < .32 else (" " + c + " ") if r < .42 else c

def mangle_gender(g):
    opts = {"male": ["male", "Male", "M", "m", "MALE"], "female": ["female", "Female", "F", "f", "FEMALE"]}[g]
    return rng.choice(opts)

# ---------------- products ----------------
write("products.csv", ["product_id", "product_name", "category", "price_inr"],
      [[f"P{i+1:03d}", n, c, p] for i, (n, c, p) in enumerate(PRODUCTS)])

# ---------------- customers ----------------
N_CUST = 1200
IMPOSSIBLE_AGE = {10: 280, 110: -4, 210: 0, 310: 150, 410: 999, 510: 200}
cust, arche = [], {}
for i in range(N_CUST):
    cid = f"C{i+1:04d}"
    r = rng.random()
    arche[cid] = "premium" if r < .20 else "frequent" if r < .60 else "rare"
    signup = datetime(2024, 1, 1) + timedelta(days=rng.randint(0, 455))
    age = IMPOSSIBLE_AGE.get(i, rng.randint(18, 60))
    age = "" if (i not in IMPOSSIBLE_AGE and rng.random() < .05) else age
    gender = "" if rng.random() < .03 else mangle_gender(rng.choice(["male", "female"]))
    cust.append([cid, mangle_city(rng.choice(CITIES)), signup.strftime("%Y-%m-%d"), age, gender])
write("customers.csv", ["customer_id", "city", "signup_date", "age", "gender"], cust)

# ---------------- orders ----------------
WIN_START, WIN_DAYS = datetime(2025, 4, 1), 183          # 2025-04-01 .. 2025-09-30
RATE = {"premium": 1.0, "frequent": 3.5, "rare": 0.5}   # expected orders per month
def pick_product(a):
    if a == "premium":
        pool = CAT_IDX["Electronics"] if rng.random() < .7 else CAT_IDX["Personal Care"] + CAT_IDX["Snacks"]
        return rng.choice(pool), rng.randint(1, 2)
    if a == "frequent":
        return rng.choice(CAT_IDX["Grocery"] + CAT_IDX["Dairy"] + CAT_IDX["Snacks"]), rng.randint(1, 3)
    return rng.choice(CAT_IDX["Grocery"] + CAT_IDX["Dairy"] + CAT_IDX["Snacks"] + CAT_IDX["Personal Care"]), rng.randint(1, 2)

rows, oid = [], 100000
for cid, a in arche.items():
    for _ in range(poisson(RATE[a] * 6.0)):
        oid += 1
        pi, qty = pick_product(a)
        ts = WIN_START + timedelta(days=rng.randint(0, WIN_DAYS - 1), hours=rng.randint(7, 22),
                                   minutes=rng.randint(0, 59), seconds=rng.randint(0, 59))
        disc = rng.choice([0, 0, 0, 5, 10, 15, 20, 25, 40])
        status = "cancelled" if rng.random() < .10 else "completed"
        amount = round(PRODUCTS[pi][2] * qty * (1 - disc / 100), 2)
        if status == "cancelled":
            deliv = ""
        else:
            deliv = rng.randint(46, 75) if rng.random() < .10 else max(8, min(45, int(round(rng.gauss(24, 8)))))
            if rng.random() < .06:
                deliv = ""
        rows.append({"order_id": f"ORD-{oid}", "customer_id": cid, "product_id": f"P{pi+1:03d}", "quantity": qty,
                     "order_ts": ts, "status": status, "discount_pct": disc, "order_amount": amount,
                     "delivery_minutes": deliv})
out = []
for k, r in enumerate(rows):
    if k % 100 == 7 and r["status"] == "completed" and r["delivery_minutes"] != "":
        r["delivery_minutes"] = -10                          # impossible negative duration (planted)
    if r["status"] == "completed" and rng.random() < .01:
        r["order_amount"] = "N/A" if rng.random() < .5 else "error"   # unparseable amount (planted)
    base = [r["order_id"], r["customer_id"], r["product_id"], r["quantity"], r["order_ts"].strftime("%Y-%m-%d %H:%M:%S"),
            r["status"], r["discount_pct"], r["order_amount"], r["delivery_minutes"]]
    out.append(base)
    if rng.random() < .05:   # earlier 'pending' record of the same order (partial duplicate)
        early = list(base); early[4] = (r["order_ts"] - timedelta(minutes=rng.randint(1, 10))).strftime("%Y-%m-%d %H:%M:%S")
        early[5] = "pending"; early[8] = ""
        out.append(early)
    if rng.random() < .01:   # exact duplicate
        out.append(list(base))
rng.shuffle(out)
write("orders.csv", ["order_id", "customer_id", "product_id", "quantity", "order_ts", "status",
                     "discount_pct", "order_amount", "delivery_minutes"], out)

# ---------------- support tickets ----------------
T = {"billing": ["I was charged twice for order {o}. Please refund the extra payment. My email is {e}.",
                 "The invoice for {o} shows a wrong amount. Call me on {p} to correct the bill.",
                 "Extra money was deducted for order {o}, please check the bill. Contact {e}."],
     "delivery": ["My order {o} has not arrived and the rider is not picking up calls. Please call {p}.",
                  "Delivery for {o} is very late, it was promised in 20 minutes. Write to {e}.",
                  "The rider marked {o} as delivered but I never received the package. Call {p}."],
     "technical": ["The app crashes every time I open the cart. I tried to place {o} again and got an error. Email {e}.",
                   "I cannot login, the OTP never arrives on {p}.",
                   "The payment page shows an error and the app freezes when I apply a coupon. Order {o}."],
     "fraud": ["There is an unauthorized transaction on my card linked to order {o}. I did not place it. Call {p}.",
               "Someone hacked my account and used my wallet balance. This is fraud, contact {e}.",
               "I was charged for order {o} that I never made, it looks like fraud on my account. Call {p}."]}
order_by_cust = {}
for r in out:
    order_by_cust.setdefault(r[1], r[0])
cust_with_orders = sorted(order_by_cust)
tickets = []
cats = list(T)
for n in range(1, 157):
    cat = cats[(n - 1) % 4]
    cid = rng.choice(cust_with_orders)
    txt = rng.choice(T[cat]).format(o=order_by_cust[cid], e=f"user{n}@example.com", p=str(rng.choice("6789")) + "".join(rng.choice("0123456789") for _ in range(9)))
    tickets.append([f"T{n:04d}", cid, order_by_cust[cid], txt, cat])
FILL = "The same problem keeps happening and nobody has given me a clear answer yet. "
for n in range(157, 161):                         # 4 empty tickets
    tickets.append([f"T{n:04d}", rng.choice(cust_with_orders), "", "", "none"])
for n in range(161, 165):                         # 4 very long tickets (~520 words)
    tickets.append([f"T{n:04d}", rng.choice(cust_with_orders), "", ("My delivery was late. " + FILL * 40).strip(), "delivery"])
rng.shuffle(tickets)
write("tickets.csv", ["ticket_id", "customer_id", "order_id", "text", "true_category"], tickets)

# ---------------- policy documents ----------------
DOCS = {
"refund_policy.txt": """Refund Policy. A refund can be requested only for orders whose status is completed. Orders with status cancelled were never charged, so no refund applies to them. If a completed order is delivered in more than 45 minutes, the customer receives a full refund of the order amount. If a completed order is delivered in 45 minutes or less, no late-delivery refund is paid. Refunds up to INR 1000 are approved automatically by the support system. A refund above INR 1000 must be approved by a human support lead before money is returned. Approved refunds are credited to the original payment method within 5 working days. A refund request must be raised within 7 days of the delivery date.""",
"delivery_policy.txt": """Delivery Policy. Standard delivery is promised within 30 minutes of order confirmation in all supported cities. The supported cities are Chennai, Mumbai, Bengaluru, Delhi, Hyderabad and Pune. A delivery fee of INR 25 is charged on orders below INR 199, and delivery is free on orders of INR 199 or more. During heavy rain the promised time may extend to 60 minutes and the app shows a rain notice. A rider will attempt to contact the customer twice before marking an order as undeliverable. Electronics items are delivered only after the customer shares a one time password with the rider.""",
"payment_policy.txt": """Payment Policy. Customers can pay using UPI, debit card, credit card or cash on delivery. Cash on delivery is available only for orders up to INR 2500. If a payment fails but money is deducted, the amount is returned automatically within 48 hours. A customer who sees two charges for one order should raise a billing ticket with the order identifier. Wallet balance can be used for any order and is never refunded as cash. Coupons cannot be combined with wallet balance on the same order.""",
"privacy_policy.txt": """Privacy Policy. Customer phone numbers and email addresses are used only for order updates and support. Support tickets are processed with personal details masked before any automated tool reads them. Delivery addresses are deleted from active systems 180 days after the last order. A customer can ask for account deletion and the request is completed within 30 days. Payment card numbers are never stored by the support system.""",
"escalation_policy.txt": """Escalation Policy. A ticket about fraud or an unauthorized transaction is marked high priority and sent to the security team within 2 hours. Tickets from premium customers are answered within 1 hour, and all other tickets within 24 hours. If an automated assistant cannot resolve a ticket within 6 steps, it must hand the case to a human agent with a short summary. A human agent must call the customer back if such a high-value request is rejected."""}
for fn, body in DOCS.items():
    with open(os.path.join(POL, fn), "w", encoding="utf-8") as f:
        f.write(body + "\n")

# ---------------- policy evaluation questions ----------------
QA = [("What delivery time makes a completed order eligible for a full refund?", "refund_policy.txt", "more than 45 minutes"),
      ("Who must approve a refund above INR 1000?", "refund_policy.txt", "human support lead"),
      ("How many days do I have to raise a refund request?", "refund_policy.txt", "within 7 days"),
      ("What is the delivery fee on small orders?", "delivery_policy.txt", "INR 25"),
      ("On orders of what value is delivery free?", "delivery_policy.txt", "INR 199 or more"),
      ("Why does the electronics delivery need a one time password?", "delivery_policy.txt", "one time password"),
      ("Is cash on delivery available for a large order?", "payment_policy.txt", "INR 2500"),
      ("How soon is money returned when a payment fails but is deducted?", "payment_policy.txt", "48 hours"),
      ("Can wallet balance be refunded as cash?", "payment_policy.txt", "never refunded as cash"),
      ("When are delivery addresses deleted from active systems?", "privacy_policy.txt", "180 days"),
      ("How fast must a fraud ticket reach the security team?", "escalation_policy.txt", "within 2 hours"),
      ("After how many steps must an automated assistant hand over to a human?", "escalation_policy.txt", "6 steps"),
      ("What is the favourite colour of the company mascot?", "NONE", ""),
      ("Who won the cricket world cup final?", "NONE", "")]
with open(os.path.join(RAW, "policy_eval.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["question", "expected_doc", "expected_fact"]); w.writerows(QA)

# ---------------- agent scenarios (simulated refund requests) ----------------
from collections import Counter
cnt = Counter(r[0] for r in out)
def clean_row(r):                                    # single-record, fully valid order
    try:
        return cnt[r[0]] == 1 and r[5] in ("completed", "cancelled") and float(r[7]) > 0 and \
            (r[5] == "cancelled" or (r[8] != "" and 8 <= int(r[8]) <= 75))
    except ValueError:
        return False
ok_rows = sorted((r for r in out if clean_row(r)), key=lambda r: r[0])
def first(pred):
    return next(r for r in ok_rows if pred(r))
late_small = [r for r in ok_rows if r[5] == "completed" and int(r[8]) > 45 and float(r[7]) <= 1000][:2]
late_big = first(lambda r: r[5] == "completed" and int(r[8]) > 45 and float(r[7]) > 1000)
on_time = first(lambda r: r[5] == "completed" and int(r[8]) <= 45)
cancelled = first(lambda r: r[5] == "cancelled")
SC = []
for r in late_small:
    SC.append((r[1], f"My order {r[0]} reached me very late. I would like a refund.", "REFUND", float(r[7])))
SC.append((late_big[1], f"Order {late_big[0]} arrived far too late, please refund me.", "ESCALATE", float(late_big[7])))
SC.append((on_time[1], f"Please refund order {on_time[0]}, I did not like the delivery.", "DENIED", 0))
SC.append((cancelled[1], f"I want my money back for order {cancelled[0]}.", "DENIED", 0))
SC.append(("C0001", "Refund for order ORD-999999 please.", "NOT_FOUND", 0))
SC.append((late_small[0][1], f"[LOOP] Refund order {late_small[0][0]}.", "FALLBACK", 0))
with open(os.path.join(RAW, "agent_scenarios.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["scenario_id", "customer_id", "message", "expected_outcome", "expected_refund_inr"])
    for i, (cid, msg, oc, amt) in enumerate(SC, 1):
        w.writerow([f"S{i}", cid, msg, oc, f"{amt:.2f}"])
print("generated: customers=%d products=%d orders_raw=%d tickets=%d" % (len(cust), len(PRODUCTS), len(out), len(tickets)))
mock_llm.py
An offline stand-in for a hosted chat-completion service. chat(messages, tools=None) returns an object shaped like a hosted API response: response.choices[0].message.content, response.choices[0].message.tool_calls (each with .id, .name, .arguments), response.choices[0].finish_reason and response.usage.total_tokens. Messages are dictionaries with role and content keys. The marker word in the first (system) message selects the behaviour: CLASSIFY (Part C), POLICY ASSISTANT (Part D RAG) or AGENT (Part D agent). est_tokens(text) is the project-wide token estimator: ceil(number_of_whitespace_separated_words × 1.3). Use it wherever this brief says "tokens".

"""mock_llm.py -- deterministic, offline stand-in for a hosted LLM chat API (no key, no network, no cost).
Interface mirrors a hosted chat API:  response = chat(messages, tools=None)
    response.choices[0].message.content / .tool_calls ; response.choices[0].finish_reason ; response.usage
Three behaviours are selected by a marker word in the FIRST (system) message:
    "CLASSIFY"  -> support-ticket classifier       "POLICY ASSISTANT" -> grounded policy answerer
    "AGENT"     -> refund agent that requests tools (ReAct style)"""
import json, math, re
from types import SimpleNamespace as NS

class TransientLLMError(Exception):
    """Simulated temporary failure (timeout / server busy). Retrying the same call succeeds."""

_state = {"calls": 0, "ok": 0}      # call counters that drive the simulated misbehaviour

def reset():
    """Restart the counters. Call once before each independent batch run."""
    _state["calls"] = 0; _state["ok"] = 0

CATS = ["billing", "delivery", "technical", "fraud"]
KW = {"fraud": {"unauthorized", "fraud", "hacked", "stolen"},
      "billing": {"charged", "invoice", "bill", "refund", "deducted", "payment"},
      "technical": {"app", "crash", "login", "otp", "error", "freezes"},
      "delivery": {"arrived", "rider", "delivery", "delivered", "late"}}

def est_tokens(text):
    return math.ceil(len((text or "").split()) * 1.3)

def _words(t):
    return set(re.findall(r"[a-z]+", (t or "").lower()))

def _resp(content=None, tool_calls=None, finish="stop", messages=()):
    prompt = sum(est_tokens(m.get("content") or json.dumps(m.get("tool_calls", ""), default=str)) for m in messages)
    comp = est_tokens(content or json.dumps(tool_calls or "", default=str))
    msg = NS(role="assistant", content=content, tool_calls=[NS(**tc) for tc in (tool_calls or [])])
    return NS(choices=[NS(message=msg, finish_reason=finish)],
              usage=NS(prompt_tokens=prompt, completion_tokens=comp, total_tokens=prompt + comp))

def _classify(messages):
    system, user = messages[0]["content"], messages[-1]["content"]
    _state["calls"] += 1
    if _state["calls"] % 9 == 0:                                 # every 9th call fails once
        raise TransientLLMError("503 server busy")
    _state["ok"] += 1
    order = ["fraud", "technical", "billing", "delivery"] if "EXAMPLES:" in system else ["billing", "fraud", "technical", "delivery"]
    w = _words(user)
    cat = next((c for c in order if KW[c] & w), "technical")
    if _state["ok"] % 13 == 0:                                   # chatty, non-JSON answer
        return _resp(f"The customer appears to have a {cat}-related problem.", messages=messages)
    if _state["ok"] % 17 == 0:                                   # JSON, but category outside the allowed list
        return _resp(json.dumps({"category": "shipping", "reason": "parcel movement"}), messages=messages)
    return _resp(json.dumps({"category": cat, "reason": "keyword match"}), messages=messages)

STOP = {"the", "a", "an", "is", "of", "to", "for", "in", "on", "how", "what", "when", "who", "are", "do", "does",
        "i", "my", "be", "can", "after", "from", "it", "or", "and", "at", "by", "which", "why", "if", "than", "that"}

def _stems(t):
    return {w[:5] for w in re.findall(r"[a-z]+", (t or "").lower()) if w not in STOP}

def _policy(messages):
    """Answers with the context sentence that best matches the question (rare words weigh more)."""
    user = messages[-1]["content"]
    m = re.search(r"CONTEXT:\s*(.*?)\s*QUESTION:\s*(.*)", user, re.S)
    context, question = (m.group(1), m.group(2)) if m else ("", user)
    sents = [s.strip() for s in re.split(r"(?<=[.])\s+", re.sub(r"\[[^\]]*\.txt\]", " ", context)) if s.strip()]
    qs = _stems(question)
    df = {q: sum(q in _stems(s) for s in sents) for q in qs}
    best, best_score = None, 0.0
    for s in sents:
        sc = sum(1.0 / df[q] for q in qs & _stems(s))
        if sc > best_score:
            best, best_score = s, sc
    if best is None:
        return _resp("I could not find that information in the policy documents.", messages=messages)
    return _resp(best, messages=messages)

def _agent(messages, tools):
    user = next(m["content"] for m in messages if m["role"] == "user")
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    call = lambda n, a: _resp(None, [{"id": f"call_{len(tool_msgs)+1}", "name": n, "arguments": a}], "tool_calls", messages)
    if "[LOOP]" in user:
        return call("search_policy", {"query": "refund policy"})
    if not tool_msgs:
        m = re.search(r"ORD-\d{6}", user)
        return call("get_order", {"order_id": m.group(0)}) if m else _resp("NEEDS_INFO: please share the order id.", messages=messages)
    last = tool_msgs[-1]; data = json.loads(last["content"]); name = last["name"]
    if name == "get_order":
        if "error" in data:
            return _resp("NOT_FOUND: no such order.", messages=messages)
        return call("search_policy", {"query": "refund policy late delivery"})
    order = json.loads(next(m for m in tool_msgs if m["name"] == "get_order")["content"])
    if name == "search_policy":
        if order["status"] != "completed":
            return _resp("DENIED: only completed orders can be refunded.", messages=messages)
        return call("calculate_refund", {"order_amount": order["order_amount"], "delivery_minutes": order["delivery_minutes"]})
    if name == "calculate_refund":
        amt = data["refund_inr"]
        if amt == 0:
            return _resp("DENIED: delivery was within 45 minutes.", messages=messages)
        if amt > 1000:
            return call("escalate_to_human", {"order_id": order["order_id"], "refund_inr": amt})
        return call("issue_refund", {"order_id": order["order_id"], "refund_inr": amt})
    if name == "issue_refund":
        return _resp(f"REFUND: INR {data['refund_inr']} issued.", messages=messages)
    if name == "escalate_to_human":
        return _resp("ESCALATE: refund above INR 1000 sent to a human lead.", messages=messages)
    return _resp("DENIED: unable to process.", messages=messages)

def chat(messages, tools=None):
    head = messages[0]["content"].upper()
    if "CLASSIFY" in head:
        return _classify(messages)
    if "POLICY ASSISTANT" in head:
        return _policy(messages)
    if "AGENT" in head:
        return _agent(messages, tools)
    raise ValueError("first message must name one of: CLASSIFY, POLICY ASSISTANT, AGENT")
Required Repository Contents
The folder layout is yours to choose, except that these paths must exist exactly as written after your README's Reproduce commands have run:

Path	Produced in
generate_data.py, mock_llm.py (unmodified), requirements.txt, .gitignore, README.md	repository root
data/clean/customers.csv, data/clean/products.csv, data/clean/orders.csv	Part A
data/instamart.db (SQLite, tables customers, products, orders)	Part A
sql/queries.sql	Part A
outputs/customer_segments.csv, outputs/lapse_model.joblib, outputs/retention_list.csv	Part B
outputs/triage_results.csv, outputs/triage_budget_demo.csv	Part C
outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl, folder state/	Part D
README.md must contain these top-level sections, in order: Overview, Reproduce (numbered shell commands, from python -m venv to the last script), Part A, Part B, Part C, Part D, Limitations and Next Steps. Every number the tasks below ask you to "report" goes into the matching Part section as a table or short list. The Part D section additionally holds the seven-scenario result table, the statuses of the two stop-safely demonstrations and the list of tools executed in the crash-and-resume run (tasks 6 to 8).

Part A — Data Foundation: Clean, Store, Query (22 marks)
The raw exports are the single source of truth for everything that follows, and they are deliberately dirty. In this Part you audit them, clean them with explicit documented rules, store the clean tables in a SQLite database, and answer business questions in SQL and in pandas. Part B reads data/clean/orders.csv and data/clean/customers.csv; Part D's refund agent reads data/instamart.db. Any error left here changes every later result.

Tasks

Reproducible setup (2 marks). Create the repository on branch main. Create an isolated virtual environment (python -m venv), install your libraries with pip, and write requirements.txt with pip freeze. Add a .gitignore that lists at least .env, your virtual-environment folder and __pycache__/. Copy the two supplied files, run python generate_data.py, and write the numbered Reproduce section of the README.

Audit the raw data (3 marks). Load the three raw tables with pandas (read tickets.csv with keep_default_na=False wherever you use it later, so empty ticket texts stay empty strings). For each table report shape, data types and, for every column, the count and percentage of missing values. Then report these specific audit figures in a table:

raw orders.csv rows; number of exact duplicate rows; number of rows whose status is pending; number of distinct order_id values;
number of order_amount entries that cannot be converted to a number; number of negative delivery_minutes; number of blank delivery_minutes;
in customers.csv: number of blank age, number of age values outside 10–100, number of distinct raw city spellings, number of distinct non-blank raw gender spellings, number of blank gender.
Reference: 13,084 order rows; 119 exact duplicates; 656 pending rows; 12,309 distinct order_id; 120 non-numeric amounts; 106 negative delivery times; 2,541 blank delivery times; 61 blank ages; 6 ages outside 10–100; 24 raw city spellings; 10 raw gender spellings; 30 blank genders.

Clean customers (3 marks). (a) Standardise city by removing leading/trailing spaces and applying title case so exactly six cities remain. (b) Map every gender spelling to male, female, or unknown (blank → unknown). (c) Treat ages outside 10–100 as missing, then fill all missing ages with the median age of the customer's city, computed with groupby and transform. (d) Convert signup_date to a datetime. Reference after cleaning: cities Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; genders female 610, male 560, unknown 30; 0 missing ages and every age between 18 and 60.

Clean orders (5 marks), in this order, documenting each rule in the README:

Convert order_ts to a datetime.
Remove exact duplicate rows.
Several orders still appear more than once with different statuses (an earlier pending record and a later final record). For each order_id keep only the record with the latest order_ts.
Convert order_amount to numeric, turning unparseable entries into missing values, then fill each missing amount from the product price: order_amount = round(price_inr × quantity × (1 − discount_pct / 100), 2).
Set negative delivery_minutes to missing. For orders with status completed, fill every missing delivery_minutes with the median delivery time of that order's product category among completed orders (use groupby and transform). For orders with status cancelled, leave delivery_minutes missing — a cancelled order was never delivered, so imputing a delivery time for it would invent data.
Reference after cleaning: 12,309 rows (one per order_id); 11,110 completed and 1,199 cancelled; no pending rows; 0 missing order_amount; 0 missing delivery_minutes among completed orders and 1,199 missing among cancelled orders; the sum of order_amount over completed orders is INR 2,998,668.80.

Store and validate (2 marks). Write the three clean tables to data/clean/ as CSV files and to the SQLite database data/instamart.db as tables named customers, products, orders (for example with sqlite3.connect and DataFrame.to_sql, or an equivalent method; store order_ts as text in the form YYYY-MM-DD HH:MM:SS). Write a validation script that asserts at least eight of the reference checkpoints from tasks 3 and 4 and prints exactly ALL CHECKPOINTS PASSED when all hold.

SQL analysis (5 marks). Save seven queries in sql/queries.sql, run each from Python with pd.read_sql_query against data/instamart.db, and report the stated result for each. Unless stated otherwise, only completed orders count as revenue or spend.

Q1 — WHERE then HAVING. Keep only completed orders with order_amount > 500 (WHERE), group by customer, and keep customers whose total of those orders exceeds 8000 (HAVING). Reference: 30 customers.
Q2 — three-table join. Revenue by city and product category (join orders, customers, products). Reference: 30 rows; the 30 revenues add up to INR 2,998,668.80; the largest cell is Bengaluru / Electronics at INR 233,909.45.
Q3 — scalar subquery. Products priced above the average catalogue price. Reference: 11 products.
Q4 — LEFT JOIN. Customers who have no completed order. Reference: 25 customers.
Q5 — common table expression. One CTE computes each customer's total completed spend and a second computes the average of those totals; the main query returns customers whose total is above the average. Reference: 550 customers; average INR 2,552.06 (rounded to 2 decimals).
Q6 — ranking window functions. For each customer with completed spend, compute total spend and then RANK(), DENSE_RANK() and ROW_NUMBER() using OVER (PARTITION BY city ORDER BY total_spend DESC). Reference: the top customer per city is Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); exactly 8 rows have RANK() different from ROW_NUMBER(), and your README states in one sentence why ties cause that.
Q7 — running total. Daily completed revenue (the day is the first ten characters of order_ts) with a cumulative revenue column computed as SUM(revenue) OVER (ORDER BY day). Reference: 183 daily rows; the last cumulative value is INR 2,998,668.80.
pandas cross-check and exploratory analysis (2 marks). (a) Rebuild Q2 with pd.pivot_table(..., margins=True) and show that its grand total equals the SQL total; with groupby and named aggregation report, per city, the sum, mean and count of completed order_amount (reference: Bengaluru sum 523,643.40, mean 276.47, count 1,894; Mumbai sum 479,058.50, mean 297.00, count 1,613). (b) For completed order_amount, report mean, median, skewness, the quartiles, the 1.5×IQR fences and the number of outliers beyond each fence, then answer in your own words, in at most 200 words in total, the four questions: What is this data about? What do its distributions and statistics look like? Are there outliers or anomalies? What relationships exist between variables? For the last question use the correlation matrix of order_amount, discount_pct, quantity, delivery_minutes over completed orders and name the pair with the largest correlation in absolute value. Reference: mean 269.91 versus median 135.00 (right-skewed), skewness about 4.61, Q1 64.80, Q3 297.00, upper fence 645.30 with 897 outliers above it and none below; the correlation largest in absolute value is order_amount with discount_pct at about −0.095 (the pair order_amount and quantity is +0.087, slightly smaller in absolute value), and your answer must say that correlation of this size does not establish causation.

Acceptance criteria (Part A is complete when…)

Running your documented commands from a clean clone regenerates the raw data and the clean tables with every reference checkpoint above matched.
The validation script prints ALL CHECKPOINTS PASSED.
data/instamart.db contains the three tables, and all seven queries in sql/queries.sql run against it and return the reference results.
Every cleaning rule appears in the README with the number of rows or values it changed.
No monetary figure anywhere uses a dollar sign; all amounts are written in INR.
Part B — Customer Segments and Lapse Prediction (28 marks)
The retention team wants two things from the clean order history: named customer segments to design campaigns around, and a lapse model that scores how likely a customer is to stop ordering. This Part consumes data/clean/orders.csv and data/clean/customers.csv from Part A, and its segment file feeds Part C's ticket routing. Use only completed orders for customer behaviour unless a task says otherwise.

Tasks

Build the RFM table (3 marks). For every customer with at least one completed order, compute, as of the snapshot moment 2025-10-01 00:00:00: recency = whole days between the snapshot and the customer's most recent completed order ((snapshot − last_order_ts).dt.days); frequency = number of distinct completed orders; monetary = sum of completed order_amount. Reference: 1,175 customers; recency median 19, minimum 0, maximum 180; frequency mean 9.46, maximum 33; monetary mean 2,552.06, maximum 14,766.55.
Cluster with K-means (5 marks). Standardise the three RFM columns with StandardScaler. For every K from 2 to 8 fit KMeans(n_clusters=K, n_init=10, random_state=42) and record the inertia and the silhouette score. Present the table, read the elbow, and choose the final K from {3, 4, 5} using the elbow, the silhouette scores and business interpretability (a number that the business can act on beats a marginally better metric). Reference (tolerance ±5% on inertia, ±0.05 on silhouette): K=2: 1820.8 / 0.465; K=3: 1268.2 / 0.445; K=4: 743.1 / 0.526; K=5: 617.4 / 0.481; K=6: 503.2 / 0.482; K=7: 408.5 / 0.416; K=8: 366.5 / 0.412. Explain in two or three sentences why the inertia of K=8 is always lower than that of K=2 yet that fact alone cannot pick K.
Profile, name and project the segments (3 marks). (a) For the chosen K, report each segment's size and its mean recency, frequency and monetary values. (b) Give each segment a business name and one recommended action, and mark as Premium the segment with the highest mean monetary value (whatever K you chose). Remind the reader that cluster numbers are only labels with no ordering. Reference for K=4: sizes about 436 / 138 / 419 / 182; the 138-customer segment has mean monetary about 7,130 and mean frequency about 6.8; the 419-customer segment has mean frequency about 19.4 and mean recency about 8.8 days. (c) Apply PCA(n_components=2) to the scaled RFM matrix and report explained_variance_ratio_ (reference: about 0.634 and 0.218, total about 0.852, tolerance ±0.02). Write outputs/customer_segments.csv with columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium.
Define the lapse problem (3 marks). Use the cutoff 2025-09-01. The cohort is every customer with at least one completed order before the cutoff. The label lapsed is 1 if the customer has no completed order dated 2025-09-01 through 2025-09-30, otherwise 0. Build, using only orders dated before the cutoff plus the clean customer table, at least these features: recency (days from the cutoff to the last completed order), frequency, monetary, average discount_pct of completed orders, cancellation rate (cancelled orders ÷ all orders with a final status, before the cutoff), average delivery_minutes of completed orders, most frequent product category, city, age, gender, and tenure in days (cutoff minus signup_date). You must not use any order dated on or after the cutoff and must not use the segment labels from task 3 as features, because both would leak the answer. Split with train_test_split(test_size=0.2, stratify=y, random_state=42). Reference: 1,156 customers in the cohort; 443 lapsed (38.32%); 924 training and 232 test rows; always predicting "not lapsed" on the test set gives accuracy 61.6% and a lapsed-class F1 of 0 (pass zero_division=0 to the scikit-learn metric functions to silence the warning).
Preprocess without leakage (3 marks). One-hot encode the nominal columns (city, gender, favourite category) using k−1 columns per variable (for example OneHotEncoder(drop="first") inside the Pipeline, fitted on the training rows only; pd.get_dummies on the whole cohort would break the training-only rule), and standardise the numeric features. Every encoder and scaler must be fitted on the training rows only and merely applied to the test rows (use a scikit-learn Pipeline, or fit on train and transform both manually). Print the learned mean of frequency from your fitted scaler next to the full-cohort mean of frequency and state in one sentence why they differ. Confirm in code that no feature column contains a missing value.
Train, tune and evaluate (6 marks). (a) Report the baseline above. (b) Tune LogisticRegression(max_iter=1000) with GridSearchCV over C ∈ {0.01, 0.1, 1, 10, 100}, cv=5, scoring="f1". (c) Tune RandomForestClassifier(n_estimators=100, random_state=42) with GridSearchCV over max_depth ∈ {3, 5, 8} and min_samples_leaf ∈ {5, 10, 20}, cv=5, scoring="f1" (45 fits). Report both best parameter sets and best cross-validated F1. (d) On the untouched test set report, for both tuned models, the confusion matrix and the precision, recall and F1 of the lapsed class, then pick one model and justify the choice in three to five sentences that mention the cost of missing a lapsing customer versus wrongly targeting a loyal one. The test set may be used for this evaluation only after tuning is finished. Reference (your numbers may differ because you may add features; they must be of this order): logistic regression best C=1, test F1 about 0.62; random forest best depth 8 and leaf 10, test F1 about 0.65; the selected model must reach a test F1 of at least 0.55 on the lapsed class.
Class-balance experiment (2 marks). The lapsed class is the minority. Over-sample the training set only: keep every original training row and add randomly chosen duplicates of lapsed training rows (drawn with replacement, random_state=42) until both classes have equal counts (570 each in the reference split), refit your best logistic regression on that balanced data, and compare precision, recall and F1 on the original, unchanged test set with the un-balanced model. State which metric moved most and what that means for the retention team. Reference: recall rises from about 0.64 to about 0.85 while precision stays near 0.6 (it does not rise).
Interpret, persist and use (3 marks). (a) Report the five most important features of your best random forest (feature_importances_) and the sign of the coefficient of three features in your best logistic regression, with a one-line business reading each. (b) Save your selected model with joblib.dump to outputs/lapse_model.joblib, reload it with joblib.load, and show that the reloaded model reproduces the original test predictions exactly. (c) Among test-set customers who are in the Premium segment, rank by the model's predicted lapse probability and write the top 10 (or all, if fewer than 10) to outputs/retention_list.csv with columns customer_id, lapse_probability, actually_lapsed; report how many of them truly lapsed. The Premium flag comes from the whole-window RFM of task 3, which includes the September window that defines the lapse label, so it is a post-hoc whole-window label: state in one sentence, both here and in the README's Limitations and Next Steps section, that the retention list is illustrative and not a leakage-free evaluation. (The model's own features still use only orders before the cutoff, as task 4 requires.)
Acceptance criteria (Part B is complete when…)

RFM, K-means table, segment profile, PCA ratios, cohort counts and split sizes match the references within the stated tolerances.
Segment assignment, scaling, encoding and model fitting are reproducible with the fixed seeds and the documented commands.
Both grid searches are reported with the stated grids; the final model reaches test F1 ≥ 0.55; metrics are computed only for the lapsed (positive) class and compared against the baseline.
outputs/customer_segments.csv, outputs/lapse_model.joblib and outputs/retention_list.csv exist with the stated columns and the Premium flag is true for exactly the highest-mean-monetary segment.
The README states explicitly that no encoder, scaler or resampler saw test rows and that no data from 2025-09-01 onwards entered the features, and that the segment labels from task 3 were not used as features; its Limitations section states that the Premium flag is a post-hoc whole-window label, so the retention list is illustrative.
Part C — Support Ticket Triage with Guardrails (22 marks)
data/raw/tickets.csv holds 164 support tickets (ticket_id, customer_id, order_id, text, true_category). Four tickets are empty and four are extremely long. Most contain an email address, a phone number or an order id that must never reach an outside model. Your job is to build a triage pipeline around the supplied mock_llm.py that is safe (privacy and input guardrails), resilient (retries), checked (output validation), cost-aware (token budget) and useful (routing that uses the customer segments from Part B). Process tickets in the order they appear in the file.

Tasks

Input guardrails (3 marks). Before any model call: (a) a ticket whose text is empty or only whitespace gets status skipped_empty and is never sent to the model; (b) a ticket whose est_tokens(text) is greater than MAX_INPUT_TOKENS = 400 gets status flagged_long and is never sent to the model. Reference: 4 empty tickets and 4 long tickets; the long ones estimate to 734 tokens.
Redact personal information (3 marks). Apply, in this order, exactly these regular expressions and placeholders to the ticket text before it goes into a prompt: email [\w.+-]+@[\w-]+\.[\w.]+ → [EMAIL]; phone \b[6-9]\d{9}\b → [PHONE]; order id ORD-\d{6} → [ORDER_ID]. Report the number of substitutions per type and in total across all 164 tickets, and show by code that re-applying the three patterns to the redacted texts finds nothing. Reference: 55 emails, 91 phone numbers, 132 order ids, 278 in total; 0 left afterwards. The content of the user message you send must be exactly the redacted ticket text and nothing else.
Two prompts (2 marks). Write a zero-shot and a few-shot system prompt. Both must contain the word CLASSIFY, list the four allowed categories (billing, delivery, technical, fraud) and require a JSON-only reply of the form {"category": "...", "reason": "..."}. The zero-shot prompt must not contain the text EXAMPLES:; the few-shot prompt must contain a line EXAMPLES: followed by at least two labelled examples, one of which is a fraud example.
Resilient calls (4 marks). Some calls to the mock raise TransientLLMError. Wrap every model call in a retry function with a maximum of 3 retries per ticket, exponential backoff with jitter — the wait before retry number n (n = 0, 1, 2) is BASE_DELAY × 2**n × random.uniform(0.9, 1.15) with BASE_DELAY = 0.05 seconds — and one log line per retry showing the ticket id, the retry number and the wait. Call mock_llm.reset() immediately before each full run. Reference: a full run over the 156 non-empty, non-long tickets produces exactly 19 TransientLLMErrors, all recovered by a retry; no ticket may end with a permanent error.
Validate every answer (4 marks). Parse the reply as JSON and accept it if category is one of the four allowed values (status classified_json). If it is not valid JSON, extract the lowercase words with re.findall(r"[a-z]+", text.lower()) (so that billing-related yields the words billing and related) and accept it if exactly one allowed category word is present among them (status classified_keyword). In every other case — including valid JSON whose category is outside the list — set status flagged_invalid_output and route the ticket to human review; never guess. Reference for one full run over the 156 eligible tickets: 135 classified_json, 12 classified_keyword, 9 flagged_invalid_output. The 12 keyword replies are the mock's chatty answers of the form The customer appears to have a <category>-related problem.; a plain whitespace split would not separate billing-related into words and would give 135 / 0 / 21 instead, so use the re.findall extraction above.
Compare zero-shot and few-shot (3 marks). Run the full pipeline once with each prompt (no token budget). For each run report: tickets classified, accuracy against true_category over the classified tickets, total tokens (sum of response.usage.total_tokens) and tokens per classified ticket. Show a confusion table for the zero-shot run, and explain in three or four sentences, with numbers from your own runs, when the extra few-shot tokens are worth paying for. Reference: 147 classified tickets (135 classified_json + 12 classified_keyword, using the word extraction of task 5); zero-shot accuracy 85.7% (126 of 147; every error is a fraud ticket predicted billing (11) or a technical ticket predicted billing (10)); few-shot accuracy 100%.
Budget, routing and output (3 marks). (a) Add a TOKEN_BUDGET: before each model call, if the running total of usage.total_tokens is already ≥ the budget, make no call and set status budget_exhausted (route to human review). Run the zero-shot pipeline with TOKEN_BUDGET = 6000 and save the result as outputs/triage_budget_demo.csv; at least one ticket must end budget_exhausted, and you report how many. If your prompt is so short that no ticket is exhausted, lower the demo budget (for example to 5000) and report the budget you used. The count depends on your prompt length (for example 51 at about 29 estimated tokens and 58 at about 33); only "at least one ticket exhausted and a reported count" is graded. (b) Produce the final outputs/triage_results.csv from a few-shot run with TOKEN_BUDGET = 100000 using this routing, which reads outputs/customer_segments.csv from Part B: tickets with status skipped_empty, flagged_long, flagged_invalid_output or budget_exhausted → route human_review, priority NORMAL; otherwise if the category is fraud → route security_team, priority HIGH; otherwise if the customer's is_premium is true → route premium_desk, priority HIGH; otherwise → route support_queue, priority NORMAL. A customer who has no row in the segment file is treated as not premium. Columns: ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used (category and segment_name left blank when not available).
Acceptance criteria (Part C is complete when…)

No empty or long ticket triggers a model call; no email, phone or order id appears in any prompt sent to the model (shown by an assertion in your code).
The retry log shows one line per retry with its wait inside the stated jitter range for that retry number (with this mock every retry is retry number 0, so each wait lies between 0.045 and 0.0575 seconds), and the 19 reference failures are all recovered.
The status counts 135 / 12 / 9 are reproduced for a full run, and the zero-shot and few-shot accuracies are 85.7% and 100% (±0.5 percentage point).
outputs/triage_budget_demo.csv shows the budget stopping the run; outputs/triage_results.csv has 164 rows and routing consistent with the stated rules.
The README explains the cost of the few-shot prompt in tokens per ticket and states clearly which outputs were rule-validated and which went to humans.
Part D — Policy RAG Assistant and Refund Agent (28 marks)
Support agents lose time looking up policies, and simple refund requests wait in a queue. In this Part you build (1) a retrieval-grounded policy assistant over the five documents in data/policies/, and (2) a refund agent that decides refund requests by calling tools: it looks up the order in data/instamart.db (Part A), searches the policies with your retrieval, computes the refund and either issues it or escalates it to a human. The agent reuses the retrieval code built in tasks 1–3. All model calls use mock_llm.py.

Refund rules the agent's tools must implement. Only completed orders can be refunded; a completed order delivered in more than 45 minutes is refunded 100% of its order_amount; delivered in 45 minutes or less → refund 0; a refund above INR 1000 is escalated to a human and not issued automatically. All simulated requests are treated as raised inside the policy's 7-day window.

Tasks

Chunk the documents (2 marks). Read the five .txt files and split each into chunks of 40 words with an overlap of 10 words: the first chunk covers words 0–39, the next starts 30 words later, and so on; stop after the first chunk whose end reaches or passes the last word of the document. Record the source file name with every chunk. Reference: 15 chunks in total — refund 4, delivery 3, escalation 3, payment 3, privacy 2.
Vectorise and retrieve (3 marks). Fit scikit-learn's TfidfVectorizer() (sklearn.feature_extraction.text) on the chunk texts, transform a query with the same fitted vectorizer, and rank chunks by cosine_similarity (sklearn.metrics.pairwise). Implement retrieve(query, k=3) returning a list of dictionaries with source, score, text. Explain in two sentences what TF and IDF each measure and one limitation of TF-IDF for this task (for example, two phrasings with the same meaning but different words). Reference: for the query refund policy late delivery the top three chunks all come from refund_policy.txt.
Augment and answer (3 marks). Write answer(question): retrieve three chunks, then send two messages to mock_llm.chat — a system message containing the words POLICY ASSISTANT that instructs the model to answer only from the supplied context and to say it could not find the information otherwise, and a user message of exactly the form CONTEXT: newline, the three chunks each prefixed with [<source file name>] on its own line, newline QUESTION: newline, the question. Return the model's text and the sources used.
Evaluate retrieval and grounding (3 marks). Run data/raw/policy_eval.csv (14 questions: 12 answerable, 2 not covered by any document). Report in a table: for the 12 answerable questions, how many have the expected_doc among the top three retrieved sources and how many answers contain expected_fact (case-sensitive substring); for the 2 unanswerable questions, how many get the fallback answer that begins I could not find. Reference: 12/12, 12/12 and 2/2. Then repeat the whole evaluation with zero overlap (40-word chunks, overlap 0), report the results, and name at least one question that breaks and explain why (an answer split across a chunk boundary). Reference: with overlap 0 at least one answerable question fails.
Define tools and the tool map (4 marks). Implement five tools, each with a name, a one-sentence description and a JSON-style parameter specification kept in a TOOLS_SPEC list, plus a dictionary TOOL_MAP from tool name to Python function: get_order(order_id) — reads data/instamart.db and returns {"order_id", "status", "order_amount", "delivery_minutes"} or {"error": "order not found"}; search_policy(query) — returns your retrieve output under the key results; calculate_refund(order_amount, delivery_minutes) — returns {"refund_inr": <amount>} following the refund rules; issue_refund(order_id, refund_inr) and escalate_to_human(order_id, refund_inr) — each append one JSON line to outputs/refund_ledger.jsonl (with a timestamp and the tool name) and return {"refund_inr": <amount>}. The model asks for tools by name with an arguments dictionary; your code looks the name up in TOOL_MAP and calls the function with those arguments.
Run the ReAct loop (5 marks). Implement run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10). Messages start as a system message containing the word AGENT and the user message. In each iteration call chat; if the reply has no tool calls, its content is the final answer; otherwise append an assistant message {"role": "assistant", "content": None, "tool_calls": [{"id", "name", "arguments"}]}, execute the tool, and append a tool message {"role": "tool", "tool_call_id", "name", "content": <JSON string of the result>}, then loop. Return a dictionary with status, final_answer, steps, llm_calls and tokens_used. The normal status is the word before the colon in the final answer (REFUND, ESCALATE, DENIED, NOT_FOUND). Run all 7 rows of data/raw/agent_scenarios.csv and append one JSON line per run to outputs/agent_runs.jsonl containing scenario_id, status, final_answer, llm_calls, tokens_used and every step's tool name, arguments and observation. Your README's Part D section must contain a small table of the seven scenario results (scenario id, status, refund amount, model calls, tokens used). With the default token_limit=6000, S7 must finish all six model calls (it uses roughly 2,700 to 5,500 estimated tokens, depending on your prompt length and how you serialise tool observations). Reference: S1 REFUND 128.25, S2 REFUND 135.00, S3 ESCALATE 1999.00, S4 DENIED, S5 DENIED, S6 NOT_FOUND, S7 falls back (task 7); the refund amount in your result must equal expected_refund_inr within 0.01, and S1 takes 5 model calls.
Stop safely (3 marks). Enforce three limits and a graceful fallback. (a) After max_iterations model calls without a final answer, stop with status FALLBACK_MAX_ITERATIONS — scenario S7 must end this way. (b) After each model call add usage.total_tokens to tokens_used; if it exceeds token_limit, stop immediately (without executing the tool just requested) with status FALLBACK_TOKEN_LIMIT; show a run of S1 with token_limit=200 ending this way (the default token_limit=6000 must not stop S7 early). (c) At the top of each iteration, if time_limit_seconds or more of wall-clock time has elapsed (elapsed >= time_limit_seconds, with elapsed measured by time.monotonic() since the run started), stop with FALLBACK_TIMEOUT; show a run with time_limit_seconds=0 ending this way (it stops before the first model call, on every platform). Every fallback returns the steps completed so far and a short human-handoff summary string in final_answer. Record the statuses of the S1 token_limit=200 demo and the time_limit_seconds=0 demo in your README's Part D section.
Make the run resumable (3 marks). After every tool observation is appended, write a checkpoint file in state/ named after a run id, containing at least the run id, scenario id, the full message list, tokens_used, the step number and the list of tools already executed. Add resume_agent(run_id) that loads the checkpoint and continues. Demonstrate with S1: raise a simulated crash right after the second tool execution (after its checkpoint is saved), then call resume_agent. The tools executed across both parts of the run must be exactly get_order, search_policy, calculate_refund, issue_refund — four executions, none repeated — and the final status must be REFUND with 128.25. List the tools executed across the crash and the resume, in order, in your README's Part D section.
ReAct versus plan-and-execute (2 marks). In at most 250 words, using the model-call counts you measured (for example 5 for S1, 3 for S5, 2 for S6), compare your ReAct agent with a plan-and-execute design that asks the model for the whole step list once and replans only on error. Cover at least: number of model calls and tokens, handling of an unexpected observation such as NOT_FOUND or a cancelled order, and which design you would choose for this refund workflow and why.
Acceptance criteria (Part D is complete when…)

Chunk counts, retrieval, grounding and fallback results match the references; the zero-overlap run is reported and at least one question fails.
Each of the seven scenarios yields the expected outcome and refund amount; S7 ends with FALLBACK_MAX_ITERATIONS under the default token_limit=6000; the token-limit and timeout demonstrations end with their own fallback statuses.
The README's Part D section contains the table of the seven scenario results, the statuses of the two demonstrations and the list of tools executed in the crash-and-resume run.
outputs/agent_runs.jsonl has one line per run with scenario_id, status, final_answer, llm_calls, tokens_used and every step; outputs/refund_ledger.jsonl contains the issued and escalated actions; state/ holds checkpoints.
The crash-and-resume demonstration shows exactly four tool executions for S1 and the correct final result.
Neither a real network call nor an API key is used anywhere.
Submission
Submit one public GitHub repository link, pointing to branch main, for the whole project. The repository must contain: the two unmodified supplied files, requirements.txt, .gitignore, all source code for Parts A–D, sql/queries.sql, the committed outputs listed in Required Repository Contents, and a single README.md with the sections Overview, Reproduce, Part A, Part B, Part C, Part D, Limitations and Next Steps and every reported number. All work must be on the single branch main; a link to any other branch, or a link that does not clearly resolve to a branch, will not be graded. Do not submit images, screenshots, PDFs, slide decks, presentations, video or audio; the entire project is text and code in the repository.