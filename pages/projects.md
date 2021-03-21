---
layout: page
title: "Projects"
description: various projects
---

<!-- {% assign posts = site.posts | where: 'category', 'projects' %}

{% for post in site.posts %}
    <div class="post-preview">
        <a href="{{ post.url | prepend: site.baseurl }}">
            <div class="row">
                <div class="col-sm-4">
                    <img style="max-height: 200px" src="/{% if post.header-img %}{{ post.header-img }}{% else %}{{ site.header-img }}{% endif %}">
                </div> 
                <div class="col-sm-8">
                    <h2 class="post-title">
                        {{ post.title }}
                    </h2>
                    {% if post.subtitle %}
                        {{ post.subtitle }}
                    {% endif %}
                </div>
            </div>
        </a>
    </div>
    {% unless forloop.last %}
        <hr>
    {% endunless %}
{% endfor %} -->

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