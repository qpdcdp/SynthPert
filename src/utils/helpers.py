import re


# --- Helper Function to Parse Answer ---
def extract_answer(generated_text):
    match = re.search(r"<answer>(.*?)</answer>", generated_text, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip().lower()
        if answer in ["upregulated", "downregulated", "not differentially expressed"]: return answer
        else:
            if "upregulated" in answer: return "upregulated"
            if "downregulated" in answer: return "downregulated"
            if "not differentially expressed" in answer: return "not differentially expressed"
            return None
    return None
