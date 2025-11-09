from langsmith import Client, wrappers
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, CONCISENESS_PROMPT
from openai import OpenAI
from langsmith.utils import LangSmithNotFoundError

client = Client()

try:
    dataset = client.read_dataset(dataset_name="ds-blank-existence-97")

except LangSmithNotFoundError:
    dataset = client.create_dataset(
        dataset_name="ds-blank-existence-97", description="A sample dataset in LangSmith."
    )
    examples = [
        {
            "inputs": {"question": "Analyse the latest UK EV subsidies vs petrol/diesel"},
            "outputs": {"answer": "should include comparison tables of petrol vs diesel EV subsidies."},
        },
        {
            "inputs": {"question": "Provide a comparative analysis of renewable energy subsidies in Germany vs France for 2025"},
            "outputs": {"answer": "Should contrast subsidy types, fiscal amounts (approx), eligibility criteria, recent policy changes."},
        },
        {
            "inputs": {"question": "Identify OWASP Top 10 2025-relevant risks for a React + Node.js ecommerce app and propose mitigations"},
            "outputs": {"answer": "Should list each risk, short description, concrete mitigation actions, prioritization."},
        },
        {
            "inputs": {"question": "Summarize new GDPR enforcement trends and notable fines in 2025"},
            "outputs": {"answer": "Should mention major cases, fine ranges, enforcement focus areas (data minimization, AI transparency)."},
        },
        # Expected to be challenging / partially failing until security context features added
        {
            "inputs": {"question": "Produce a detailed risk assessment for migrating a legacy monolith to microservices (security & compliance)"},
            "outputs": {"answer": "Should include threat surface changes, auth/identity implications, data residency, logging gaps."},
        },
        {
            "inputs": {"question": "Audit the security configuration of our Kubernetes cluster (no config details provided)"},
            "outputs": {"answer": "Should explain inability to audit without config; list required artifacts and a high-level checklist."},
        },
        {
            "inputs": {"question": "Generate a penetration testing plan for internal systems with unspecified architecture"},
            "outputs": {"answer": "Should state missing architecture details; outline ethical scope boundaries, phases, required approvals."},
        },
        {
            "inputs": {"question": "Evaluate potential risks of using deprecated TLS versions and recommend migration steps"},
            "outputs": {"answer": "Should list deprecated versions, risks (MITM, weak ciphers), upgrade path, testing/rollback plan."},
        },
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)

# Wrap the OpenAI client for LangSmith tracing
openai_client = wrappers.wrap_openai(OpenAI())

# Define the application logic to evaluate.
# Dataset inputs are automatically sent to this target function.
def target(inputs: dict) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-5-chat-latest",
        messages=[
            {"role": "system", "content": "Answer the following question accurately"},
            {"role": "user", "content": inputs["question"]},
        ],
    )
    return {"answer": response.choices[0].message.content}

# Define an LLM-as-a-judge evaluator to evaluate correctness of the output
def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="gpt-5-chat-latest",
        feedback_key="correctness",
    )
    eval_result = evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    )
    return eval_result

def conciseness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CONCISENESS_PROMPT,
        model="gpt-5-chat-latest",
        feedback_key="conciseness",
    )
    eval_result = evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    )
    return eval_result

experiment_results = client.evaluate(
    target,
    data="ds-blank-existence-97",
    evaluators=[correctness_evaluator, conciseness_evaluator],
    experiment_prefix="experiment-quickstart-jaunty-frenzy-96",
    max_concurrency=2,
)