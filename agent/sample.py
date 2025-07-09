dic = {
    "role": "tool",
    "content": "Hello",
    "tool_calls": [
        {
            "tool_name": "render_bar_chart",
            "tool_arguments": {"topic": "Amazon Revenue (Last 4 Years)", "data": [{"x": "2021", "y": 469822000000}, ...]}
        }
    ]
}
del dic['role']
print(dic)