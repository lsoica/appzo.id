---
layout: category
title: "ai"
---

<ul>
  {% for post in site.categories.ai %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <small>{{ post.date | date: "%B %d, %Y" }}</small>
    </li>
  {% endfor %}
</ul>
