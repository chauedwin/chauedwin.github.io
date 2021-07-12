---
layout: page
description: Personal Page 
---

{% for post in site.posts %}
  <div class="post-list">
    <h2>
      <a href="{{ post.url }}">
        {{ post.title }}
      </a>
    </h2>
	{{ post.preview }} 
	<br>
    <time datetime="{{ post.date | date: "%Y-%m-%d" }}">{{ post.date | date_to_long_string }}</time>
    
  </div>
{% endfor %}