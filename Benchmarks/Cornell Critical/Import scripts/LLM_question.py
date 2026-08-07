import ollama
import json
import matplotlib.pyplot as plt
import math
from collections import Counter
from random import choice


class LLM:
    """
    Spør en lokal Ollama-modell om spørsmål fra en JSON-fil på formatet:

    {
      "items": [
        {
          "item_id": <int>,
          "text": <str>,
          "options": [{"label": <str>, "text": <str>}, ...],
          "correct_label": <str>,
          ... (ekstra felt, f.eks. "group"/"form"/"question_word_count", ignoreres)
        },
        ...
      ]
    }

    item_id trenger IKKE være sammenhengende eller starte på 1 (f.eks. Cornell CRT
    bruker 7-78), og antall svaralternativer trenger IKKE være 5 (Cornell bruker A/B/C).
    Begge deler leses fra selve dataene i stedet for å være hardkodet.
    """

    def __init__(self, model: str, options: dict, rel_path: str, prompt_system: str, show_progress: bool = True):
        self.model = model
        self.options = options
        self.rel_path = rel_path
        self.prompt_system = prompt_system
        self._show_progress = show_progress

    # ── Hjelpemetoder ──────────────────────────────────────────────

    def load_items(self, question_file: str) -> list[dict]:
        """Leser og returnerer listen av spørsmål fra en JSON-fil (rel_path + question_file + '.json')."""
        with open(self.rel_path + question_file + ".json", encoding="utf-8") as f:
            return json.load(f)["items"]

    @staticmethod
    def _build_prompt(question: dict) -> str:
        text = question["text"] + "\n"
        for option in question["options"]:
            text += f"{option['label']}) {option['text']}\n"
        return text + "\n"

    @staticmethod
    def _valid_letters(question: dict) -> set[str]:
        """Gyldige svarbokstaver for DETTE spørsmålet, hentet fra options i stedet for en fast global mengde."""
        return {option["label"] for option in question["options"]}

    @staticmethod
    def _parse_letter(raw: str) -> str:
        """
        Henter ut svarbokstaven fra modellens rå tekstrespons.
        Tar bare det første tegnet (stort bokstav), slik at ting som
        "A)", "A." eller "A:" fortsatt godtas som "A" - modeller har en
        tendens til å gjenta formatet fra promptet ("A) Yes") selv om de
        er bedt om å svare med kun bokstaven.
        """
        raw = raw.strip().upper()
        return raw[0] if raw else ""

    def _ask_once(self, prompt: str, want_logprobs: bool = False):
        """Spør modellen én gang og returnerer hele completion-objektet."""
        return ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt_system},
                {"role": "user", "content": prompt},
            ],
            options=self.options,
            logprobs=want_logprobs,
            top_logprobs=5 if want_logprobs else None,
        )

    @staticmethod
    def _extract_probs(completion: dict):
        """Henter ut sannsynlighet per bokstav fra logprobs."""
        probs = {}
        for token_set in completion.get("logprobs", []):
            for cand in token_set["top_logprobs"]:
                probs[cand["token"]] = math.exp(cand["logprob"])
        return probs

    # ── Hoved-API ──────────────────────────────────────────────────
    # Alle tre metodene returnerer nå dict[item_id -> verdi] i stedet for
    # faste 30-lange lister indeksert på item_id - 1. Det gjør dem trygge å
    # bruke uansett hvor mange spørsmål filen har, eller hvilken item_id
    # nummerering den starter på.

    def ask_model(self, question_file: str, questions_to_answer: list[int] | None = None):
        """
        JSON av spørsmål -> (answers_letter, answers_probability), begge dict[item_id -> verdi].
        questions_to_answer=None betyr "svar på alle spørsmål i filen".
        """
        answers_letter = {}
        answers_probability = {}

        for q in self.load_items(question_file):
            if questions_to_answer is not None and q["item_id"] not in questions_to_answer:
                continue

            prompt = self._build_prompt(q)
            if self._show_progress:
                print(f"{question_file} spørsmål {q['item_id']}")
            comp = self._ask_once(prompt, want_logprobs=True)
            answer = self._parse_letter(comp["message"]["content"])
            valid = self._valid_letters(q)
            if answer not in valid:
                raise ValueError(f'Ugyldig svar "{answer}" på spørsmål {q["item_id"]} (gyldige: {sorted(valid)}, rått svar: "{comp["message"]["content"]!r}")')
            answers_letter[q["item_id"]] = answer
            answers_probability[q["item_id"]] = self._extract_probs(comp)

        return answers_letter, answers_probability

    def ask_model_n_times(self, question_file: str, n: int, questions_to_answer: list[int] | None = None):
        """
        JSON av spørsmål -> (answers_letter, answers_probability), begge dict[item_id -> verdi].
        Bruker majoritetsvalg over n spørringer per spørsmål.
        """
        answers_letter = {}
        answers_probability = {}

        for q in self.load_items(question_file):
            if questions_to_answer is not None and q["item_id"] not in questions_to_answer:
                continue
            prompt = self._build_prompt(q)
            valid = self._valid_letters(q)
            votes = []
            for i in range(n):
                if self._show_progress:
                    print(f"{question_file} spørsmål {q['item_id']} gang {i + 1}")
                comp = self._ask_once(prompt)
                vote = self._parse_letter(comp["message"]["content"])
                if vote not in valid:
                    raise ValueError(f'Ugyldig svar "{vote}" på spørsmål {q["item_id"]} (gyldige: {sorted(valid)}, rått svar: "{comp["message"]["content"]!r}")')
                votes.append(vote)
            count = Counter(votes)
            most = max(count.values())
            winners = [l for l, c in count.items() if c == most]
            answers_letter[q["item_id"]] = choice(winners)
            answers_probability[q["item_id"]] = {l: c / n for l, c in count.items()}

        return answers_letter, answers_probability

    def ask_model_with_explanations(self, question_file: str, questions_to_answer: list[int] | None = None):
        """
        Går gjennom alle spørsmålene ett og ett, spør modellen, og
        returnerer (answers_letter, answers_explanation), begge dict[item_id -> verdi].
        Forventer at modellen svarer med bokstaven på første linje og
        forklaringen på linjene under.
        """
        answers_letter = {}
        answers_explanation = {}

        for q in self.load_items(question_file):
            if questions_to_answer is not None and q["item_id"] not in questions_to_answer:
                continue
            prompt = self._build_prompt(q)
            if self._show_progress:
                print(f"{question_file} spørsmål {q['item_id']}")
            comp = self._ask_once(prompt)
            raw = comp["message"]["content"].strip()
            lines = raw.splitlines()
            letter = self._parse_letter(lines[0]) if lines else ""
            explanation = "\n".join(lines[1:]).strip()
            valid = self._valid_letters(q)
            if letter not in valid:
                raise ValueError(f'Ugyldig svar "{letter}" på spørsmål {q["item_id"]} (gyldige: {sorted(valid)}, rått svar: "{raw!r}")')
            answers_letter[q["item_id"]] = letter
            answers_explanation[q["item_id"]] = explanation

        return answers_letter, answers_explanation


# ── Tilhørende funksjoner ────────────────────────────────────────────
# Disse tar nå `items` (listen fra LLM.load_items(...)) som argument i
# stedet for å lese en separat answers_true.txt. Fasit (correct_label) og
# alternativer (options) hentes direkte fra JSON-filen, så det er én
# kilde til sannhet i stedet for to filer som må holdes synkronisert.

def plot_distributions(
    answers_probability: dict[int, dict[str, float]],
    items: list[dict],
    plot_title: str = "Probability distributions from all questions",
    save: bool = False,
    filename: str = "distribution_plot.png",
    path: str = "",
) -> None:
    """
    Plotter sannsynlighetsdistribusjonene hentet ut før softmax om save er False.
    Om save er True, lagres plottet i stedet.
    """
    correct_by_id = {q["item_id"]: q["correct_label"] for q in items}
    labels_by_id = {q["item_id"]: [opt["label"] for opt in q["options"]] for q in items}

    item_ids = sorted(answers_probability.keys())
    number_of_questions = len(item_ids)
    number_of_columns = 5
    number_of_rows = math.ceil(number_of_questions / number_of_columns)

    fig, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(15, number_of_rows * 3),
        squeeze=False,
    )
    axes = axes.flatten()

    correct_count = 0
    for c, item_id in enumerate(item_ids):
        ax = axes[c]
        probabilities = answers_probability[item_id]
        answer_labels = labels_by_id.get(item_id, ["A", "B", "C", "D", "E"])

        percentages = [probabilities.get(label, 0) * 100 for label in answer_labels]

        true_lbl = correct_by_id.get(item_id)
        if true_lbl in answer_labels:
            true_idx = answer_labels.index(true_lbl)
            ax.axvline(x=true_idx, color="red", linewidth=1)

            pred_idx = max(range(len(percentages)), key=percentages.__getitem__)
            if answer_labels[pred_idx] == true_lbl:
                correct_count += 1

        ax.bar(answer_labels, percentages)
        ax.set_title(f"Question {item_id}", pad=15, fontsize=9)
        ax.set_ylabel("Probability (%)", fontsize=8)
        ax.set_ylim(0, 100)

        for label_index, percentage in enumerate(percentages):
            ax.text(
                label_index,
                percentage + 3,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    # Fjerner tomme plott
    for unused_index in range(number_of_questions, len(axes)):
        axes[unused_index].set_visible(False)

    fig.suptitle(plot_title, fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(hspace=0.6, wspace=0.4, bottom=0.05)
    fig.text(0.99, 0.01, f"Correct: {correct_count}/{number_of_questions}", ha="right", va="bottom", fontsize=8)

    if save:
        plt.savefig(path + filename)
    else:
        plt.show()


def print_explanation(
    answers_letter: dict[int, str],
    answers_explanation: dict[int, str],
    items: list[dict],
) -> None:
    """Printer modellens svar, fasit og forklaring for hvert besvarte spørsmål."""
    correct_by_id = {q["item_id"]: q["correct_label"] for q in items}

    for item_id in sorted(answers_explanation.keys()):
        true_letter = correct_by_id.get(item_id)
        print(f"{item_id}. Answered: {answers_letter[item_id]}, correct answer: {true_letter}")
        print(answers_explanation[item_id])
        print("\n")


# ── Eksempel på bruk (Cornell reasoning-filen) ───────────────────────
#
# llm = LLM(
#     model="llama3",
#     options={"temperature": 0},
#     rel_path="edited_json_files/",          # mappa der cornell_reasoning_items.json ligger
#     prompt_system="Svar kun med bokstaven til riktig alternativ.",
# )
# items = llm.load_items("cornell_reasoning_items")
# answers_letter, answers_probability = llm.ask_model("cornell_reasoning_items")
# plot_distributions(answers_probability, items, plot_title="Cornell CRT — llama3")