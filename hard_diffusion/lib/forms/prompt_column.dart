import 'package:flutter/material.dart';
import 'package:hard_diffusion/forms/fields/prompts.dart';
import 'package:hard_diffusion/forms/fields/toggle_switch.dart';
import 'package:hard_diffusion/state/app.dart';
import 'package:provider/provider.dart';

class PromptColumn extends StatelessWidget {
  const PromptColumn({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    var appState = context.watch<MyAppState>();
    var channel = appState.channel;
    var connected = appState.webSocketConnected;
    var prompt = appState.prompt;
    var negativePrompt = appState.negativePrompt;
    return Flexible(
      child: ListView(
        shrinkWrap: true,
        padding: EdgeInsets.all(15.0),
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Column(
              children: [
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Text("Generate",
                      style: Theme.of(context).textTheme.titleLarge),
                ),
                PromptField(promptValue: prompt, setPrompt: appState.setPrompt),
                NegativePromptField(
                    negativePromptValue: negativePrompt,
                    setNegativePrompt: appState.setNegativePrompt),
                ToggleSwitch(
                    setValue: appState.setUseAdvanced, label: "Advanced"),
                ToggleSwitch(
                    setValue: appState.setUsePreview, label: "Preview"),
                ToggleSwitch(setValue: appState.setUseNsfw, label: "NSFW"),
                ElevatedButton(
                    onPressed: appState.generate, child: Text('Generate')),
                Column(
                  children: [
                    if (connected && channel != null) ...[
                      Text("Connected"),
                    ] else ...[
                      Text("Not connected"),
                    ],
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
