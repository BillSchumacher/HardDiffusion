import 'package:flutter/material.dart';
import 'package:hard_diffusion/forms/fields/field_constants.dart';

class IntegerInputField extends StatelessWidget {
  const IntegerInputField(
      {super.key,
      required this.label,
      required this.setValue,
      required this.value});

  final Function(int) setValue;
  final int value;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      child: Flexible(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(label),
            Padding(
              padding: inputPadding,
              child: TextFormField(
                onSaved: (newValue) => setValue(int.parse(newValue!)),
                initialValue: value.toString(),
                decoration: inputDecoration,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
