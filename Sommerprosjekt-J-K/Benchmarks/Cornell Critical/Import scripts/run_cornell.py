import os
from LLM_question import LLM, plot_distributions


def run_item_bank(llm: LLM, file_name: str, output_dir: str) -> None:
    """Kjører llm på ett item bank-fil og lagrer et distribusjonsplott i output_dir."""
    try:
        items = llm.load_items(file_name)
    except FileNotFoundError:
        print(f"Hopper over {file_name}: fant ikke {llm.rel_path}{file_name}.json")
        return

    print(f"Spør {llm.model} om {file_name} ({len(items)} spørsmål)...")
    answers_letter, answers_probability = llm.ask_model(file_name)

    plot_distributions(
        answers_probability,
        items,
        plot_title=file_name,
        save=True,
        filename=file_name + ".png",
        path=output_dir,
    )


def run_all(llm: LLM, filenames: list[str], output_dir: str = "figures/") -> None:
    """Kjører llm på en liste med item bank-filer og lagrer ett plott per fil."""
    os.makedirs(output_dir, exist_ok=True)
    for file_name in filenames:
        run_item_bank(llm, file_name, output_dir)


if __name__ == "__main__":
    model_name = "llama3.1:8b"

    ollama_options = {
        "temperature": 0,     # 0 gir mer deterministiske svar
        "seed": 42,           # fast seed + temperature=0 -> deterministisk
        "num_predict": 5,   # sett lavere (f.eks. 5) hvis du ikke ber om begrunnelse
        "num_ctx": 512,
    }

    prompt_system = (
        "Answer the multiple choice question, answer in exactly this format:\n"
        "ONLY the letter (A-E)\n"
        # "A short explanation of why this is correct\n"
    )

    Qwen = LLM(model_name, ollama_options, "edited_json_files/", prompt_system)

    # Bytt ut med hvilke som helst item bank-filnavn (uten .json), f.eks.
    # ["cornell_reasoning_items"] eller ["sci_item_bank", "gmat_items"].
    filenames = [
        "cardinal_list", "cardinal",
        "cartesian_list", "cartesian",
        "clock_face_list", "clock_face",
        "descriptive_context_list_nuc", "descriptive_context_list",
        "descriptive_context_nuc", "descriptive_context",
    ]

    run_all(Qwen, filenames, output_dir="figures/")