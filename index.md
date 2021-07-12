---
layout: page
description: Personal Page 
---

## Interesting Posts:

{% for post in site.categories.interesting %}
  <div class="post-list">
    <h2>
      <a href="{{ post.url }}">
        {{ post.title }}
      </a>
    </h2>
	{{ post.preview }}
	<br>
    
  </div>
{% endfor %}

<br>

## Some of my artwork!


<p align="center">
  <img src="{{site.baseurl}}/img/hmc.jpg" height=400px>
</p>

<br>

<p align="center">
  <img src="{{site.baseurl}}/img/guilin.jpg" height=600px>
</p>