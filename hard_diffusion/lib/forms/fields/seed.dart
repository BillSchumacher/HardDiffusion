import 'package:flutter/material.dart';

class SeedField extends StatelessWidget {
  const SeedField({super.key, required this.setValue, required this.value});

  final Function(int) setValue;
  final int value;

  @override
  Widget build(BuildContext context) {
    return Padding(
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
    );
  }
}
