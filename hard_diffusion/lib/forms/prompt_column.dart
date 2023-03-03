import 'package:flutter/material.dart';
import 'package:hard_diffusion/forms/fields/advanced_switch.dart';
import 'package:hard_diffusion/forms/fields/nsfw_switch.dart';
import 'package:hard_diffusion/forms/fields/prompts.dart';
import 'package:hard_diffusion/main.dart';
import 'package:provider/provider.dart';

class PromptColumn extends StatelessWidget {
  const PromptColumn({
    super.key,
    required this.setPrompt,
    required this.setNegativePrompt,
    required this.setUseAdvanced,
    required this.generate,
    required this.prompt,
    required this.negativePrompt,
  });

  final Function(String) setPrompt;
  final Function(String) setNegativePrompt;
  final Function(bool) setUseAdvanced;
  final Function() generate;
  final String prompt;
  final String negativePrompt;

  @override
  Widget build(BuildContext context) {
    var appState = context.watch<MyAppState>();
    var channel = appState.channel;
    var connected = appState.webSocketConnected;
    return Flexible(
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text("Generate",
                  style: Theme.of(context).textTheme.titleLarge),
            ),
            PromptField(promptValue: prompt, setPrompt: setPrompt),
            NegativePromptField(
                negativePromptValue: negativePrompt,
                setNegativePrompt: setNegativePrompt),
            AdvancedSwitch(setValue: setUseAdvanced),
            NSFWSwitch(),
            ElevatedButton(onPressed: generate, child: Text('Generate')),
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
    );
  }
}
