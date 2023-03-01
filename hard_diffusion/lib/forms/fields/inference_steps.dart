import 'package:flutter/material.dart';

class InferenceStepsField extends StatelessWidget {
  const InferenceStepsField({
    super.key,required this.setValue, required this.value});

  final Function(int) setValue;
  final int value;

  @override
  Widget build(BuildContext context) {
    return Flexible(
      child: Column(
        children: [
          Text("Inference Steps:"),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: TextFormField(
              onSaved: (newValue) => setValue(int.parse(newValue!)),
              initialValue: value.toString(),
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
