import 'package:flutter/material.dart';

class WidthField extends StatelessWidget {
  const WidthField({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Flexible(
      child: Column(
        children: [
          Text("Width:"),
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
