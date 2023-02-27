import 'package:flutter/material.dart';

class SeedField extends StatelessWidget {
  const SeedField({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: TextField(
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
