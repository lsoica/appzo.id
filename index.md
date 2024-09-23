---
layout: sidebar-layout
title: Appzoid
---

# Welcome to Appzoid Blog

Hello and welcome to Appzoid blog!

Feel free to follow me on:
- [Twitter](https://twitter.com/lsoica)
- [LinkedIn](https://linkedin.com/in/lsoi)
- [GitHub](https://github.com/lsoica)

## Latest Posts

<ul>
  {% for post in site.posts limit:5 %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <small>{{ post.date | date: "%B %d, %Y" }}</small>
    </li>
  {% endfor %}
</ul>


## Subscribe

<form action="https://example.us10.list-manage.com/subscribe/post" method="post">
  <input type="email" name="EMAIL" placeholder="Enter your email" required>
  <button type="submit">Subscribe</button>
</form>
