import 'package:flutter/material.dart';
import 'package:hard_diffusion/forms/fields/field_constants.dart';

class FloatInputField extends StatelessWidget {
  const FloatInputField(
      {super.key,
      required this.label,
      required this.setValue,
      required this.value});

  final Function(double) setValue;
  final double value;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Flexible(
      child: Column(
        children: [
          Text(label),
          Padding(
            padding: inputPadding,
            child: TextFormField(
                onSaved: (newValue) => setValue(double.parse(newValue!)),
                initialValue: value.toString(),
                decoration: inputDecoration),
          ),
        ],
      ),
    );
  }
}
