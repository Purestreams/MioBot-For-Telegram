from typing import Optional

from app.ai_model import chat_completion_text


async def plain_text_to_markdown(text: str, *, model: Optional[str] = None) -> str:
    """Convert plain text into markdown using the configured LLM provider."""
    prompt = f"""
        You are an expert technical writer specializing in markdown formatting. Your task is to convert the following plain text into a readable markdown document.

        **Instructions:**

        1.  **Structure:**
            *   Use `#` for the main title, `##` for major headings, and `###` for subheadings to create a clear document hierarchy.

        2.  **Emphasis:**
            *   Use **bold** (`**text**`) for key terms and important phrases.
            *   Use *italics* (`*text*`) for emphasis or to define terms.

        3.  **Code:**
            *   Use backticks (`` `code` ``) for inline code snippets, commands, or file names.
            *   Use triple backticks (``````) for multi-line code blocks, and specify the programming language if it's apparent (e.g., ```python).

        4.  **Other Elements:**
            *   Use blockquotes (`>`) for quotations or special notes.
            *   Format any data that appears to be tabular into a markdown table.

        **Constraint:**
        *   Keep all the sentences in the original text.
        *   Do not make extra headings or subheadings that are not present in the original text.
        *   Keep the paragraphs and line breaks as they are in the original text.
        *   Keep the structure of the sentences generally the same as the original text.
        *   Only use bullets and numbers for lists if they are present in the original text.
        *   Do not add any additional information or context that is not present in the original text.
        *   Do not alter the original meaning of the text.
        *   Do not write any comments in the markdown which are not present in the original text.
        *   Provide only the formatted markdown in your response.

        **Plain Text to Convert:**
        '''
        {text}
        '''

        """

    markdown_content = await chat_completion_text(
        messages=[
            {"role": "system", "content": "You are a markdown formatting expert."},
            {"role": "user", "content": prompt},
        ],
        model=model,
    )
    return markdown_content.strip()


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        sample_text = (
            'A story about China 1000 years ago, a person died. This is the end of the story. '
            'python code: print("hello world") c++ code: stdout >> "c++ output" Appendix The university of China'
        )
        markdown = await plain_text_to_markdown(sample_text)
        print(markdown)

    asyncio.run(main())

