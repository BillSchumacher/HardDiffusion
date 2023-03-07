import 'package:hard_diffusion/forms/advanced_column.dart';
import 'package:hard_diffusion/forms/prompt_column.dart';
import 'package:flutter/material.dart';
import 'package:hard_diffusion/state/app.dart';
import 'package:provider/provider.dart';

class InferTextToImageForm extends StatelessWidget {
  const InferTextToImageForm({super.key, required this.landscape});
  final bool landscape;
  @override
  Widget build(BuildContext context) {
    var appState = context.watch<MyAppState>();
    var useAdvanced = appState.useAdvanced;
    if (landscape) {
      return Form(
        key: appState.inferenceFormKey,
        child: Row(
          children: [
            Flexible(
              child: Container(
                child: ListView(shrinkWrap: true, children: [
                  //Container(child: AdvancedColumn()),
                  Container(child: PromptForm(landscape: landscape)),
                ]),
              ),
            ),
          ],
        ),
      );
    }
    return Form(
      key: appState.inferenceFormKey,
      child: Flexible(
        child: Row(
          children: [
            Flexible(
              child: ListView(children: [
                PromptForm(
                  landscape: landscape,
                ),
                /*if (useAdvanced) ...[
                AdvancedColumn(),
                VerticalSeparator(),
                PromptColumn(),
              ] else ...[
                PromptColumn(),
              ]*/
              ]),
            ),
          ],
        ),
      ),
    );
  }
}
