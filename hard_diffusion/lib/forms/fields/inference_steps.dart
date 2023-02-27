import 'package:flutter/material.dart';

class InferenceStepsField extends StatelessWidget {
  const InferenceStepsField({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Flexible(
      child: Column(
        children: [
          Text("Inference Steps:"),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: TextField(
              decoration: InputDecoration(
                border: OutlineInputBorder(),
                filled: true,
                fillColor: Colors.white,
                hintText: '',
              ),
            ),
          ),
        ],
      ),
    );
  }
}
