import 'package:flutter/material.dart';
import 'package:hard_diffusion/forms/fields/field_constants.dart';

class PromptField extends StatelessWidget {
  const PromptField(
      {super.key,
      required this.landscape,
      required this.setPrompt,
      required this.promptValue});

  final bool landscape;
  final Function(String) setPrompt;
  final String promptValue;

  @override
  Widget build(BuildContext context) {
    var lines = 4;
    if (landscape) {
      lines = 2;
    }
    return Padding(
      padding: const EdgeInsets.all(4.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text("Prompt:"),
          TextFormField(
            onSaved: (newValue) => setPrompt(newValue!),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Please enter some text';
              }
              return null;
            },
            initialValue: promptValue,
            maxLines: lines,
            decoration: InputDecoration(
              border: OutlineInputBorder(),
              filled: true,
              fillColor: Colors.white,
              hintText: 'An astronaut riding on a horse.',
            ),
          )
        ],
      ),
    );
  }
}

class NegativePromptField extends StatelessWidget {
  const NegativePromptField(
      {super.key,
      required this.landscape,
      required this.negativePromptValue,
      required this.setNegativePrompt});

  final bool landscape;
  final Function(String) setNegativePrompt;
  final String negativePromptValue;

  @override
  Widget build(BuildContext context) {
    var lines = 3;
    if (landscape) {
      lines = 2;
    }
    return Padding(
      padding: const EdgeInsets.all(4.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text("Negative Prompt:"),
          TextFormField(
            onSaved: (newValue) => setNegativePrompt(newValue!),
            validator: (value) {
              return null;
            },
            initialValue: negativePromptValue,
            maxLines: lines,
            decoration: inputDecoration,
          )
        ],
      ),
    );
  }
}
