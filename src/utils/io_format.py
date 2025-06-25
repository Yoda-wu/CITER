import re


def small_model_extract_answer(response):
    match = re.search(r"\(([a-zA-Z])\)", response)
    if match:
        return match.group(1).upper()
    else:
        return "NULL"


def large_model_extract_answer(answer, choices, labels):
    answer_label = re.search(r"###\s*(.{0,150})", answer)
    if answer_label:
        answer_text = answer_label.group(1).strip()
        for label, choice in zip(labels, choices):
            if choice.lower() in answer_text.lower():
                return label, answer_text
    return "NULL", "NULL"


def mmlu_extract_answer(answer):
    answer_label = re.search(r"###\s{0,5}([A-D])\.", answer)
    if answer_label:
        return answer_label.group(1)
    return "NULL"


def route_extract_answer(response, choices, labels):
    # 1: text after “Therefore”
    therefore_matches = re.finditer(r"Therefore(.{0,50})", response)
    for match in therefore_matches:
        answer_text = match.group(1).strip()
        for label, choice in zip(labels, choices):
            if choice.lower() in answer_text.lower():
                return label, choice

    # 2: standalone uppercase letter after "Therefore" A/A./(A)
    therefore_matches = re.finditer(r"Therefore(.{0,50})", response)
    for match in therefore_matches:
        answer_text = match.group(1).strip()
        capital_letter_match = re.search(
            r"[\(\[“‘\"\'\s]*([A-Z])[\)\]”’\"\'\.\,\;\:\?\!]*\b(?![a-zA-Z])",
            answer_text,
        )
        if capital_letter_match:
            detected_letter = capital_letter_match.group(1)
            return detected_letter.upper(), "detected by label"

    # 3. letter inside parentheses (A)/(a)
    bracket_match = re.search(r"\(([a-zA-Z])\)", response)
    if bracket_match:
        detected_letter = bracket_match.group(1)
        return detected_letter.upper(), "detected by bracket"

    # 4. standalone uppercase letter within first 10 chars
    initial_match = re.search(
        r"^[\s\(\[“‘\"\'\n]*([A-Z])[\)\]”’\"\'\.\,\;\:\?\!]*\b(?![a-zA-Z])",
        response[:15],
    )
    if initial_match:
        detected_letter = initial_match.group(1)
        return detected_letter.upper(), "detected in initial match"

    # 5. choice text within first 20 chars
    answer_start = response[:20]
    for label, choice in zip(labels, choices):
        if choice.lower() in answer_start.lower():
            return label, "detected by start 20 characters"

    return "NULL", "NULL"


def generate_large_prompt(question, labels, choices):
    prompt = f"Question: {question}\nChoices:\n"
    for label, choice in zip(labels, choices):
        prompt += f"{choice}\n"
    prompt += (
        "Answer the question and provide an explanation. The correct answer is: ###"
    )
    return prompt


def generate_prompt(question, labels, choices):
    prompt = f"Question: {question}\nChoices:\n"
    for label, choice in zip(labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += (
        "Please format your response in the following way:\n"
        "[Explanation]. Therefore, the answer is answer (label).\n"
        "Ensure the final sentence includes the answer followed by the label in parentheses.\n"
        "Answer and Reasoning:"
    )
    return prompt


def generate_math_prompt(question):
    prompt = "Solve the following math problem step by step. Ensure all reasoning is shown clearly in LaTeX format, and the final answer should be enclosed using the LaTeX command \\boxed{} to display it in a box.\n"
    prompt += f"Question: {question}\n"
    prompt += "Solution: "
    return prompt


def extract_math_answer(solution):
    start = solution.find(r"\boxed{")
    if start == -1:
        return None

    start += len(r"\boxed{")

    open_braces = 1
    end = start

    while end < len(solution) and open_braces > 0:
        if solution[end] == "{":
            open_braces += 1
        elif solution[end] == "}":
            open_braces -= 1
        end += 1

    return solution[start : end - 1]
