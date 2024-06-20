from shiny import App, render, ui, reactive
from seinGPT import MakeSeinfeldModel, generateOutput, encodeTokens, decodeTokens
from pathlib import Path

root = Path(__file__).parent

app_ui = ui.page_fluid(
        ui.layout_columns(
            ui.panel_title("Seinfeld GPT"),
            ui.input_dark_mode(mode="dark"),
        ),
        ui.card(
            "",
            ui.layout_columns(
                ui.input_text("inputTokens", "Token Input", ""),
                ui.input_slider("tokenCount", "Tokens to generate", 1, 500, 20),
            ),
        ),
        ui.input_action_button("generate", "Generate"),
        ui.output_text_verbatim("outputText"),
        ui.output_text_verbatim("footText"),
        ui.output_text_verbatim("generatorStart"),
        ui.output_text_verbatim("generatorUpdate")
)


def server(input, output, session):
    tokens = reactive.value("")
    textHolder = reactive.value("")
    tokensToGen = reactive.value(0)
    genStatus = reactive.value("")
    path = root.__str__() + "\seinfeld_CPU.pth"
    model = MakeSeinfeldModel(path)

    @render.text
    def outputText():
        if tokensToGen.get() <= 0:
            genStatus.set("Done!!!")
            return
        newTokens = tokens.get()
        if len(newTokens) <= 1:
            newTokens = '\n'
        newTokens = encodeTokens(newTokens)
        tCount = input.tokenCount()
        newTokens = generateOutput(model, newTokens)
        newTokens = decodeTokens(newTokens)

        textHolder.set(newTokens)

        ui.update_slider("tokenCount", value=tCount - 1)
        return newTokens

    @render.text
    @reactive.event(input.generate)
    def generatorStart():
        genStatus.set("Generating...")
        startingTokens = input.inputTokens()
        if len(startingTokens) <= 1:
            startingTokens = '\n'
        tokens.set(input.inputTokens())
        tokensToGen.set(input.tokenCount())
        return ""
    
    @render.text
    @reactive.event(input.tokenCount)
    def generatorUpdate():
        tokens.set(textHolder.get())
        tokensToGen.set(tokensToGen.get() - 1)
        return ""
    
    @render.text
    def footText():
        status = genStatus.get()
        return status

app = App(app_ui, server)
