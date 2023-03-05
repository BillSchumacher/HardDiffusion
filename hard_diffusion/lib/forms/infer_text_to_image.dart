import 'package:hard_diffusion/forms/advanced_column.dart';
import 'package:hard_diffusion/forms/prompt_column.dart';
import 'package:flutter/material.dart';
import 'package:hard_diffusion/state/app.dart';
import 'package:hard_diffusion/vertical_separator.dart';
import 'package:provider/provider.dart';

class InferTextToImageForm extends StatelessWidget {
  const InferTextToImageForm({super.key});

  @override
  Widget build(BuildContext context) {
    var appState = context.watch<MyAppState>();
    var useAdvanced = appState.useAdvanced;

    return Form(
      key: appState.inferenceFormKey,
      child: Flexible(
        child: Row(children: [
          if (useAdvanced) ...[
            AdvancedColumn(),
            VerticalSeparator(),
            PromptColumn(),
          ] else ...[
            PromptColumn(),
          ]
        ]),
      ),
    );
  }
}
