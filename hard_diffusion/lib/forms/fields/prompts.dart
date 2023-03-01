import 'package:flutter/material.dart';

class PromptField extends StatelessWidget {
  const PromptField(
      {super.key, required this.setPrompt, required this.promptValue});

  final Function(String) setPrompt;
  final String promptValue;
  
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(8.0),
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
            maxLines: 4,
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
      required this.negativePromptValue,
      required this.setNegativePrompt});

  final Function(String) setNegativePrompt;
  final String negativePromptValue;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(8.0),
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
            maxLines: 4,
            decoration: InputDecoration(
              border: OutlineInputBorder(),
              filled: true,
              fillColor: Colors.white,
              hintText: '',
            ),
          )
        ],
      ),
    );
  }
}
